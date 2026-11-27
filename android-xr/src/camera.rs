//! Quest passthrough camera, through the Camera2 NDK.
//!
//! Meta exposes the forward-facing cameras on Quest 3 and 3S as ordinary
//! Camera2 devices, distinguished from the avatar cameras by vendor tags.
//! Their own documentation covers the Kotlin path; this is the native one,
//! because everything else here is Rust and bouncing frames through JNI to
//! get them back into a Vulkan pipeline would be absurd.
//!
//! Requirements, all of which fail silently if missed:
//!
//! * `horizonos.permission.HEADSET_CAMERA` **granted at runtime**. Declaring
//!   it is not enough. For a POC:
//!   `adb shell pm grant rust.dinovision_xr horizonos.permission.HEADSET_CAMERA`
//! * minSdk 34 and NDK r27 for the API-34 sysroot. Note that Horizon OS
//!   v205 does *not* export `ACameraManager_getTagFromName`, so Meta's
//!   vendor tags cannot be resolved by name; cameras are identified by
//!   lens facing and supported formats instead.
//! * Passthrough enabled on the device.
//!
//! The camera delivers 1280×960 YUV420 at 60 Hz. Frames are passed on in
//! that layout, repacked to tight I420: the grade needs only the Y plane,
//! and model preprocessing happens on the GPU.
//!
//! Ported from DinoVision's `camera.rs`, which is where the device
//! knowledge above was earned; keep the two in sync when either learns
//! something new about Horizon OS.

#![cfg(target_os = "android")]

use std::ffi::{c_char, c_int, c_void};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use red_eye::frame::{Eye, Frame};
use red_eye::source::FrameSource;

/// How long to wait between reopen attempts. The camera stays disabled for
/// as long as the headset is off, so retrying hard would just spin.
const REOPEN_INTERVAL: Duration = Duration::from_secs(1);

// --- Opaque handles -------------------------------------------------------

#[repr(C)]
struct ACameraManager(#[allow(dead_code)] [u8; 0]);
#[repr(C)]
struct ACameraDevice(#[allow(dead_code)] [u8; 0]);
#[repr(C)]
struct ACameraMetadata(#[allow(dead_code)] [u8; 0]);
#[repr(C)]
struct ACameraCaptureSession(#[allow(dead_code)] [u8; 0]);
#[repr(C)]
struct ACaptureRequest(#[allow(dead_code)] [u8; 0]);
#[repr(C)]
struct ACameraOutputTarget(#[allow(dead_code)] [u8; 0]);
#[repr(C)]
struct ACaptureSessionOutput(#[allow(dead_code)] [u8; 0]);
#[repr(C)]
struct ACaptureSessionOutputContainer(#[allow(dead_code)] [u8; 0]);
#[repr(C)]
struct AImageReader(#[allow(dead_code)] [u8; 0]);
#[repr(C)]
struct AImage(#[allow(dead_code)] [u8; 0]);
#[repr(C)]
struct ANativeWindow(#[allow(dead_code)] [u8; 0]);

#[repr(C)]
struct ACameraIdList {
    num_cameras: c_int,
    camera_ids: *mut *const c_char,
}

#[repr(C)]
struct ACameraMetadataConstEntry {
    tag: u32,
    kind: u8,
    count: u32,
    data: *const u8,
}

#[repr(C)]
struct ACameraDeviceStateCallbacks {
    context: *mut c_void,
    on_disconnected: extern "C" fn(*mut c_void, *mut ACameraDevice),
    on_error: extern "C" fn(*mut c_void, *mut ACameraDevice, c_int),
}

#[repr(C)]
struct ACameraCaptureSessionStateCallbacks {
    context: *mut c_void,
    on_closed: extern "C" fn(*mut c_void, *mut ACameraCaptureSession),
    on_ready: extern "C" fn(*mut c_void, *mut ACameraCaptureSession),
    on_active: extern "C" fn(*mut c_void, *mut ACameraCaptureSession),
}

const AIMAGE_FORMAT_YUV_420_888: c_int = 0x23;
const TEMPLATE_PREVIEW: c_int = 1;
const ACAMERA_OK: c_int = 0;
const AMEDIA_OK: c_int = 0;

/// Standard metadata tags, section ordinal `<< 16` plus index.
///
/// Meta's vendor tags would be nicer, but resolving a vendor tag by name
/// needs `ACameraManager_getTagFromName`, and Horizon OS v205 does not
/// export it — linking against it stops the whole library from loading
/// with `UnsatisfiedLinkError`. Standard tags are all that is portable
/// here, so the passthrough cameras get identified by what they can do
/// rather than by what they are called.
const ACAMERA_LENS_FACING: u32 = (8 << 16) + 5;
const ACAMERA_SCALER_AVAILABLE_STREAM_CONFIGURATIONS: u32 = (13 << 16) + 10;
const ACAMERA_LENS_POSE_ROTATION: u32 = (8 << 16) + 6;
const ACAMERA_LENS_POSE_TRANSLATION: u32 = (8 << 16) + 7;
const ACAMERA_LENS_INTRINSIC_CALIBRATION: u32 = (8 << 16) + 10;
const ACAMERA_LENS_POSE_REFERENCE: u32 = (8 << 16) + 12;
const ACAMERA_LENS_DISTORTION: u32 = (8 << 16) + 13;
const ACAMERA_SENSOR_INFO_ACTIVE_ARRAY_SIZE: u32 = 15 << 16;
const ACAMERA_SENSOR_INFO_PRE_CORRECTION_ACTIVE_ARRAY_SIZE: u32 = (15 << 16) + 10;

/// The camera's optics and its place on the headset, from the standard
/// Camera2 characteristics, where Horizon OS fills them. Each is `None`
/// when the camera does not report it.
#[derive(Clone, Debug, Default)]
pub struct Calibration {
    /// `fx, fy, cx, cy, skew`, in pixels of the pre-correction active array.
    pub intrinsics: Option<[f32; 5]>,
    /// Radial `k1..k3` and tangential `p1, p2`.
    pub distortion: Option<[f32; 5]>,
    /// The lens's orientation, a quaternion `x, y, z, w`, relative to
    /// `pose_reference`.
    pub rotation: Option<[f32; 4]>,
    /// The lens's position in metres, relative to `pose_reference`.
    pub translation: Option<[f32; 3]>,
    /// `LENS_POSE_REFERENCE`: 0 primary camera, 1 gyroscope, 2 undefined,
    /// 3 automotive.
    pub pose_reference: Option<u8>,
    /// `left, top, width, height` of the active array.
    pub active_array: Option<[i32; 4]>,
    /// The same before distortion correction, which the intrinsics use.
    pub pre_correction_array: Option<[i32; 4]>,
}

impl Calibration {
    /// Read what `md` reports.
    unsafe fn read(md: *const ACameraMetadata) -> Self {
        unsafe fn values<T: Copy, const N: usize>(
            md: *const ACameraMetadata,
            tag: u32,
        ) -> Option<[T; N]> {
            unsafe {
                let mut entry = ACameraMetadataConstEntry {
                    tag: 0,
                    kind: 0,
                    count: 0,
                    data: std::ptr::null(),
                };
                if ACameraMetadata_getConstEntry(md, tag, &mut entry) != ACAMERA_OK
                    || entry.data.is_null()
                    || (entry.count as usize) < N
                {
                    return None;
                }
                let data = std::slice::from_raw_parts(entry.data as *const T, N);
                Some(std::array::from_fn(|i| data[i]))
            }
        }
        unsafe {
            Self {
                intrinsics: values(md, ACAMERA_LENS_INTRINSIC_CALIBRATION),
                distortion: values(md, ACAMERA_LENS_DISTORTION),
                rotation: values(md, ACAMERA_LENS_POSE_ROTATION),
                translation: values(md, ACAMERA_LENS_POSE_TRANSLATION),
                pose_reference: values::<u8, 1>(md, ACAMERA_LENS_POSE_REFERENCE).map(|[r]| r),
                active_array: values(md, ACAMERA_SENSOR_INFO_ACTIVE_ARRAY_SIZE),
                pre_correction_array: values(
                    md,
                    ACAMERA_SENSOR_INFO_PRE_CORRECTION_ACTIVE_ARRAY_SIZE,
                ),
            }
        }
    }
}

/// The passthrough cameras look outward at the world, so they report
/// `LENS_FACING_BACK`. The Quest also exposes a front-facing camera —
/// the avatar one — which advertises the same resolutions but is
/// separately restricted and denies `openCamera` outright.
const LENS_FACING_BACK: u8 = 1;

/// Camera2 error codes, so a failure says something rather than showing a
/// bare negative number.
fn camera_error(code: c_int) -> &'static str {
    match code {
        -10001 => "invalid parameter",
        -10002 => "camera disconnected",
        -10003 => "not enough memory",
        -10004 => "metadata not found",
        -10005 => "camera device error",
        -10006 => "camera service error",
        -10007 => "session closed",
        -10008 => "invalid operation",
        -10009 => "stream configure failed",
        -10010 => "camera in use",
        -10011 => "max cameras in use",
        -10012 => "camera disabled",
        -10013 => "permission denied",
        -10014 => "unsupported operation",
        _ => "unknown",
    }
}

#[link(name = "camera2ndk")]
unsafe extern "C" {
    fn ACameraManager_create() -> *mut ACameraManager;
    fn ACameraManager_delete(m: *mut ACameraManager);
    fn ACameraManager_getCameraIdList(
        m: *mut ACameraManager,
        out: *mut *mut ACameraIdList,
    ) -> c_int;
    fn ACameraManager_deleteCameraIdList(l: *mut ACameraIdList);
    fn ACameraManager_getCameraCharacteristics(
        m: *mut ACameraManager,
        id: *const c_char,
        out: *mut *mut ACameraMetadata,
    ) -> c_int;
    fn ACameraManager_openCamera(
        m: *mut ACameraManager,
        id: *const c_char,
        cb: *mut ACameraDeviceStateCallbacks,
        out: *mut *mut ACameraDevice,
    ) -> c_int;
    fn ACameraMetadata_getConstEntry(
        md: *const ACameraMetadata,
        tag: u32,
        entry: *mut ACameraMetadataConstEntry,
    ) -> c_int;
    fn ACameraMetadata_free(md: *mut ACameraMetadata);
    fn ACameraDevice_close(d: *mut ACameraDevice) -> c_int;
    fn ACameraDevice_createCaptureRequest(
        d: *mut ACameraDevice,
        template: c_int,
        out: *mut *mut ACaptureRequest,
    ) -> c_int;
    fn ACameraDevice_createCaptureSession(
        d: *mut ACameraDevice,
        outputs: *const ACaptureSessionOutputContainer,
        cb: *const ACameraCaptureSessionStateCallbacks,
        session: *mut *mut ACameraCaptureSession,
    ) -> c_int;
    fn ACaptureSessionOutputContainer_create(
        out: *mut *mut ACaptureSessionOutputContainer,
    ) -> c_int;
    fn ACaptureSessionOutputContainer_add(
        c: *mut ACaptureSessionOutputContainer,
        o: *const ACaptureSessionOutput,
    ) -> c_int;
    fn ACaptureSessionOutputContainer_free(c: *mut ACaptureSessionOutputContainer);
    fn ACaptureSessionOutput_create(
        w: *mut ANativeWindow,
        out: *mut *mut ACaptureSessionOutput,
    ) -> c_int;
    fn ACaptureSessionOutput_free(o: *mut ACaptureSessionOutput);
    fn ACameraOutputTarget_create(
        w: *mut ANativeWindow,
        out: *mut *mut ACameraOutputTarget,
    ) -> c_int;
    fn ACameraOutputTarget_free(t: *mut ACameraOutputTarget);
    fn ACaptureRequest_addTarget(r: *mut ACaptureRequest, t: *const ACameraOutputTarget) -> c_int;
    fn ACaptureRequest_free(r: *mut ACaptureRequest);
    fn ACameraCaptureSession_setRepeatingRequest(
        s: *mut ACameraCaptureSession,
        cb: *mut c_void,
        num: c_int,
        requests: *mut *mut ACaptureRequest,
        seq: *mut c_int,
    ) -> c_int;
    fn ACameraCaptureSession_close(s: *mut ACameraCaptureSession);
}

#[link(name = "mediandk")]
unsafe extern "C" {
    fn AImageReader_new(
        width: c_int,
        height: c_int,
        format: c_int,
        max_images: c_int,
        out: *mut *mut AImageReader,
    ) -> c_int;
    fn AImageReader_delete(r: *mut AImageReader);
    fn AImageReader_getWindow(r: *mut AImageReader, out: *mut *mut ANativeWindow) -> c_int;
    fn AImageReader_acquireLatestImage(r: *mut AImageReader, out: *mut *mut AImage) -> c_int;
    fn AImage_delete(i: *mut AImage);
    fn AImage_getWidth(i: *const AImage, out: *mut i32) -> c_int;
    fn AImage_getHeight(i: *const AImage, out: *mut i32) -> c_int;
    fn AImage_getTimestamp(i: *const AImage, out: *mut i64) -> c_int;
    fn AImage_getPlaneRowStride(i: *const AImage, plane: c_int, out: *mut i32) -> c_int;
    fn AImage_getPlanePixelStride(i: *const AImage, plane: c_int, out: *mut i32) -> c_int;
    fn AImage_getPlaneData(
        i: *const AImage,
        plane: c_int,
        data: *mut *mut u8,
        len: *mut c_int,
    ) -> c_int;
}

/// Set from the device callbacks, cleared when the camera is reopened.
///
/// A process-global rather than a per-camera flag because the callbacks
/// take a raw context pointer and this app only ever opens one camera;
/// threading an `Arc` through the FFI to support a second would be
/// ceremony for a case that does not exist.
static DEVICE_ERROR: AtomicBool = AtomicBool::new(false);

extern "C" fn on_disconnected(_ctx: *mut c_void, _d: *mut ACameraDevice) {
    log::warn!("camera disconnected");
    DEVICE_ERROR.store(true, Ordering::Release);
}

extern "C" fn on_error(_ctx: *mut c_void, _d: *mut ACameraDevice, err: c_int) {
    // Code 3 is ERROR_CAMERA_DISABLED, which Horizon OS raises whenever the
    // headset comes off. It is routine rather than exceptional here, so the
    // camera reopens instead of the view freezing on its last frame.
    log::warn!(
        "camera device error {err}{}",
        if err == 3 {
            " (disabled — headset removed?)"
        } else {
            ""
        }
    );
    DEVICE_ERROR.store(true, Ordering::Release);
}
extern "C" fn on_session(_ctx: *mut c_void, _s: *mut ACameraCaptureSession) {}

/// Index of an eye's camera among the passthrough pair, which Camera2
/// lists left then right.
fn camera_position(eye: Eye) -> u8 {
    match eye {
        Eye::Left => 0,
        Eye::Right => 1,
    }
}

/// A live passthrough camera delivering I420 frames at sensor resolution.
pub struct PassthroughCamera {
    manager: *mut ACameraManager,
    device: *mut ACameraDevice,
    session: *mut ACameraCaptureSession,
    request: *mut ACaptureRequest,
    target: *mut ACameraOutputTarget,
    output: *mut ACaptureSessionOutput,
    container: *mut ACaptureSessionOutputContainer,
    reader: *mut AImageReader,
    frame: Frame,
    have_frame: bool,
    /// The open camera's optics and pose, as far as it reports them.
    calibration: Calibration,
    /// Kept so the camera can be reopened after the device errors.
    eye: Eye,
    capture: (i32, i32),
    last_reopen: Instant,
}

// The handles are only touched from the thread that owns the struct; the
// NDK does not pin them to a thread.
unsafe impl Send for PassthroughCamera {}

impl PassthroughCamera {
    /// The open camera's optics and pose.
    pub fn calibration(&self) -> &Calibration {
        &self.calibration
    }

    /// Open the passthrough camera for one eye.
    ///
    /// `capture` is the sensor resolution to request — 1280×960 is what the
    /// Quest 3/3S offer.
    pub fn new(eye: Eye, capture: (i32, i32)) -> Result<Self, String> {
        unsafe {
            let manager = ACameraManager_create();
            if manager.is_null() {
                return Err("ACameraManager_create returned null".into());
            }

            let mut list: *mut ACameraIdList = std::ptr::null_mut();
            if ACameraManager_getCameraIdList(manager, &mut list) != ACAMERA_OK || list.is_null() {
                ACameraManager_delete(manager);
                return Err("could not enumerate cameras — is HEADSET_CAMERA granted?".into());
            }
            let ids = std::slice::from_raw_parts((*list).camera_ids, (*list).num_cameras as usize);
            log::info!("{} cameras visible", ids.len());

            let candidates = Self::candidates(manager, ids, eye, capture);
            if candidates.is_empty() {
                ACameraManager_deleteCameraIdList(list);
                ACameraManager_delete(manager);
                return Err("no outward-facing camera offers the requested format".into());
            }

            // --- Reader and its surface ---
            let mut reader: *mut AImageReader = std::ptr::null_mut();
            if AImageReader_new(
                capture.0,
                capture.1,
                AIMAGE_FORMAT_YUV_420_888,
                // A small queue: we always take the newest frame and drop
                // the rest, so depth only adds latency.
                4,
                &mut reader,
            ) != AMEDIA_OK
            {
                ACameraManager_deleteCameraIdList(list);
                ACameraManager_delete(manager);
                return Err("AImageReader_new failed".into());
            }
            let mut window: *mut ANativeWindow = std::ptr::null_mut();
            AImageReader_getWindow(reader, &mut window);

            // --- Open the device ---
            let mut callbacks = ACameraDeviceStateCallbacks {
                context: std::ptr::null_mut(),
                on_disconnected,
                on_error,
            };
            // Try each candidate in preference order. Which camera id maps
            // to which physical sensor is not documented, and some are
            // restricted in ways that only surface at open time, so trying
            // beats predicting.
            let mut device: *mut ACameraDevice = std::ptr::null_mut();
            let mut last = ACAMERA_OK;
            let mut calibration = Calibration::default();
            for &id in &candidates {
                let status = ACameraManager_openCamera(manager, id, &mut callbacks, &mut device);
                if status == ACAMERA_OK && !device.is_null() {
                    log::info!("opened camera {:?}", std::ffi::CStr::from_ptr(id));
                    let mut md: *mut ACameraMetadata = std::ptr::null_mut();
                    if ACameraManager_getCameraCharacteristics(manager, id, &mut md) == ACAMERA_OK {
                        calibration = Calibration::read(md);
                        ACameraMetadata_free(md);
                    }
                    log::info!("camera calibration: {calibration:?}");
                    break;
                }
                log::warn!(
                    "camera {:?} would not open: {} ({status})",
                    std::ffi::CStr::from_ptr(id),
                    camera_error(status)
                );
                last = status;
                device = std::ptr::null_mut();
            }
            ACameraManager_deleteCameraIdList(list);
            if device.is_null() {
                AImageReader_delete(reader);
                ACameraManager_delete(manager);
                return Err(format!(
                    "no camera would open; last error {} ({last}){}",
                    camera_error(last),
                    if last == -10013 {
                        " — HEADSET_CAMERA is a runtime permission and must be \
                         granted, and passthrough must be enabled"
                    } else {
                        ""
                    }
                ));
            }

            // --- Session and repeating request ---
            let mut container: *mut ACaptureSessionOutputContainer = std::ptr::null_mut();
            ACaptureSessionOutputContainer_create(&mut container);
            let mut output: *mut ACaptureSessionOutput = std::ptr::null_mut();
            ACaptureSessionOutput_create(window, &mut output);
            ACaptureSessionOutputContainer_add(container, output);

            let session_cb = ACameraCaptureSessionStateCallbacks {
                context: std::ptr::null_mut(),
                on_closed: on_session,
                on_ready: on_session,
                on_active: on_session,
            };
            let mut session: *mut ACameraCaptureSession = std::ptr::null_mut();
            if ACameraDevice_createCaptureSession(device, container, &session_cb, &mut session)
                != ACAMERA_OK
            {
                return Err("createCaptureSession failed".into());
            }

            let mut request: *mut ACaptureRequest = std::ptr::null_mut();
            ACameraDevice_createCaptureRequest(device, TEMPLATE_PREVIEW, &mut request);
            let mut target: *mut ACameraOutputTarget = std::ptr::null_mut();
            ACameraOutputTarget_create(window, &mut target);
            ACaptureRequest_addTarget(request, target);

            let mut requests = [request];
            if ACameraCaptureSession_setRepeatingRequest(
                session,
                std::ptr::null_mut(),
                1,
                requests.as_mut_ptr(),
                std::ptr::null_mut(),
            ) != ACAMERA_OK
            {
                return Err("setRepeatingRequest failed".into());
            }

            log::info!(
                "passthrough camera streaming {}x{} YUV420",
                capture.0,
                capture.1
            );
            Ok(Self {
                calibration,
                manager,
                device,
                session,
                request,
                target,
                output,
                container,
                reader,
                // Mid-grey until the first frame lands, so a stalled camera
                // is visibly "no data" rather than black.
                frame: {
                    let mut frame = Frame::new(capture.0 as u32, capture.1 as u32, eye);
                    frame.luma_mut().fill(128);
                    frame
                },
                have_frame: false,
                eye,
                capture,
                last_reopen: Instant::now(),
            })
        }
    }

    /// Find the passthrough camera for the requested eye.
    ///
    /// Meta identifies these with vendor tags, but resolving a vendor tag
    /// needs `ACameraManager_getTagFromName`, which Horizon OS v205 does not
    /// export — and merely *linking* it prevents the library from loading at
    /// all. So the cameras are identified by capability instead: the
    /// passthrough pair are the ones offering the requested YUV420 size.
    /// Among those, the list order is left then right, matching how Meta
    /// numbers them.
    unsafe fn candidates(
        manager: *mut ACameraManager,
        ids: &[*const c_char],
        eye: Eye,
        capture: (i32, i32),
    ) -> Vec<*const c_char> {
        unsafe {
            let mut matching = Vec::new();
            for &id in ids {
                let mut md: *mut ACameraMetadata = std::ptr::null_mut();
                if ACameraManager_getCameraCharacteristics(manager, id, &mut md) != ACAMERA_OK {
                    continue;
                }

                let mut entry = ACameraMetadataConstEntry {
                    tag: 0,
                    kind: 0,
                    count: 0,
                    data: std::ptr::null(),
                };
                let facing = (ACameraMetadata_getConstEntry(md, ACAMERA_LENS_FACING, &mut entry)
                    == ACAMERA_OK
                    && entry.count > 0)
                    .then(|| *entry.data);

                let mut supported = false;
                let mut entry = ACameraMetadataConstEntry {
                    tag: 0,
                    kind: 0,
                    count: 0,
                    data: std::ptr::null(),
                };
                if ACameraMetadata_getConstEntry(
                    md,
                    ACAMERA_SCALER_AVAILABLE_STREAM_CONFIGURATIONS,
                    &mut entry,
                ) == ACAMERA_OK
                    && !entry.data.is_null()
                {
                    // int32[n * 4]: format, width, height, is-input.
                    let values =
                        std::slice::from_raw_parts(entry.data as *const i32, entry.count as usize);
                    let (configs, _) = values.as_chunks::<4>();
                    supported = configs.iter().any(|&[format, width, height, is_input]| {
                        format == AIMAGE_FORMAT_YUV_420_888
                            && (width, height) == capture
                            && is_input == 0
                    });
                }
                ACameraMetadata_free(md);

                log::info!(
                    "camera {:?}: facing={facing:?} offers {}x{} YUV420: {supported}",
                    std::ffi::CStr::from_ptr(id),
                    capture.0,
                    capture.1
                );
                if supported {
                    matching.push((id, facing));
                }
            }

            // Outward-facing first, and within that the requested eye
            // first — but keep the rest as fallbacks, since the mapping
            // from camera id to physical sensor is undocumented.
            let (mut outward, inward): (Vec<_>, Vec<_>) = matching
                .into_iter()
                .partition(|&(_, facing)| facing == Some(LENS_FACING_BACK));

            let wanted = camera_position(eye) as usize;
            if wanted < outward.len() {
                outward.swap(0, wanted);
            }
            let ordered: Vec<*const c_char> = outward
                .into_iter()
                .chain(inward)
                .map(|(id, _)| id)
                .collect();

            if ordered.is_empty() {
                log::warn!(
                    "no camera advertises {}x{} YUV420; trying all of them",
                    capture.0,
                    capture.1
                );
                return ids.to_vec();
            }
            ordered
        }
    }

    /// Pull the newest frame, repacking it into tight I420.
    fn pump(&mut self) -> bool {
        unsafe {
            let mut image: *mut AImage = std::ptr::null_mut();
            if AImageReader_acquireLatestImage(self.reader, &mut image) != AMEDIA_OK
                || image.is_null()
            {
                return false;
            }

            let (mut w, mut h) = (0i32, 0i32);
            AImage_getWidth(image, &mut w);
            AImage_getHeight(image, &mut h);

            let plane = |index: c_int| -> Option<(*mut u8, usize, usize)> {
                let (mut data, mut len) = (std::ptr::null_mut(), 0);
                if AImage_getPlaneData(image, index, &mut data, &mut len) != AMEDIA_OK {
                    return None;
                }
                let (mut row, mut pixel) = (0i32, 0i32);
                AImage_getPlaneRowStride(image, index, &mut row);
                AImage_getPlanePixelStride(image, index, &mut pixel);
                Some((data, row as usize, pixel.max(1) as usize))
            };

            let (
                Some((y_data, y_row, _)),
                Some((u_data, u_row, u_pix)),
                Some((v_data, v_row, v_pix)),
            ) = (plane(0), plane(1), plane(2))
            else {
                AImage_delete(image);
                return false;
            };

            let mut timestamp = 0i64;
            AImage_getTimestamp(image, &mut timestamp);
            copy_yuv420(
                YuvPlanes {
                    y: y_data,
                    y_row,
                    u: u_data,
                    u_row,
                    u_pix,
                    v: v_data,
                    v_row,
                    v_pix,
                },
                w as usize,
                h as usize,
                &mut self.frame,
            );
            self.frame.timestamp_ns = timestamp;
            AImage_delete(image);
            self.have_frame = true;
            true
        }
    }
}

impl Drop for PassthroughCamera {
    fn drop(&mut self) {
        unsafe {
            if !self.session.is_null() {
                ACameraCaptureSession_close(self.session);
            }
            if !self.request.is_null() {
                ACaptureRequest_free(self.request);
            }
            if !self.target.is_null() {
                ACameraOutputTarget_free(self.target);
            }
            if !self.container.is_null() {
                ACaptureSessionOutputContainer_free(self.container);
            }
            if !self.output.is_null() {
                ACaptureSessionOutput_free(self.output);
            }
            if !self.device.is_null() {
                ACameraDevice_close(self.device);
            }
            if !self.reader.is_null() {
                AImageReader_delete(self.reader);
            }
            if !self.manager.is_null() {
                ACameraManager_delete(self.manager);
            }
        }
    }
}

impl FrameSource for PassthroughCamera {
    fn next_frame(&mut self) -> Option<&Frame> {
        // Taking the headset off disables the camera, which is a normal
        // thing to do rather than a fatal one. Reopen instead of freezing
        // on the last frame forever.
        if DEVICE_ERROR.load(Ordering::Acquire) && self.last_reopen.elapsed() >= REOPEN_INTERVAL {
            self.last_reopen = Instant::now();
            match Self::new(self.eye, self.capture) {
                Ok(mut fresh) => {
                    // Carry the last good frame across so the view does not
                    // flash grey on every reopen.
                    std::mem::swap(&mut fresh.frame, &mut self.frame);
                    fresh.have_frame = self.have_frame;
                    DEVICE_ERROR.store(false, Ordering::Release);
                    // Dropping the old value closes the previous handles.
                    *self = fresh;
                    log::info!("camera reopened");
                }
                Err(e) => log::warn!("camera reopen failed: {e}"),
            }
        }

        // `None` when nothing new arrived: the caller keeps what it has.
        (self.pump() && self.have_frame).then_some(&self.frame)
    }
}

struct YuvPlanes {
    y: *mut u8,
    y_row: usize,
    u: *mut u8,
    u_row: usize,
    u_pix: usize,
    v: *mut u8,
    v_row: usize,
    v_pix: usize,
}

/// Repack a strided, possibly semi-planar YUV420 image into `frame`'s
/// tight I420 layout. Camera2 pads rows and usually interleaves chroma
/// (pixel stride 2), so neither plane can be copied wholesale.
///
/// # Safety
///
/// The plane pointers must be valid for the given strides and dimensions.
unsafe fn copy_yuv420(p: YuvPlanes, width: usize, height: usize, frame: &mut Frame) {
    if frame.width as usize != width || frame.height as usize != height {
        *frame = Frame::new(width as u32, height as u32, frame.eye);
    }
    let luma = frame.luma_mut();
    for y in 0..height {
        let row = unsafe { std::slice::from_raw_parts(p.y.add(y * p.y_row), width) };
        luma[y * width..][..width].copy_from_slice(row);
    }
    let (cw, ch) = (width / 2, height / 2);
    let base = width * height;
    for cy in 0..ch {
        for cx in 0..cw {
            let o = base + cy * cw + cx;
            frame.data[o] = unsafe { *p.u.add(cy * p.u_row + cx * p.u_pix) };
            frame.data[o + cw * ch] = unsafe { *p.v.add(cy * p.v_row + cx * p.v_pix) };
        }
    }
}
