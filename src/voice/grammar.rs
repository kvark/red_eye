//! The command grammar, and decoding constrained to it.
//!
//! Free decoding of a short command mishears object names: on synthetic
//! speech Moonshine-tiny got only half of them, "bicycle" becoming "mysql".
//! Commands come from a small grammar, so decoding picks, at each step, the
//! likeliest token that keeps the transcript inside it, which gets nearly
//! all of them. The grammar is a trie of token ids, precomputed from the
//! tokenizer by `tools/prepare_moonshine.py`, so the app needs no
//! tokenizer.
//!
//! Constrained decoding always lands on some command, even for speech that
//! is none. Its confidence, the mean log-probability of the path under the
//! model's free distribution, tells those apart: off-grammar speech forces
//! tokens the model did not expect.

use crate::perception::Label;

/// The token that starts a transcript, and the one that ends it.
pub const START: u32 = 1;
pub const END: u32 = 2;

/// What the wearer asked for.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Command {
    /// Find and lock on to this class.
    Track(Label),
    /// Drop the target.
    Cancel,
}

#[derive(Debug, Default)]
struct Node {
    children: Vec<(u32, usize)>,
    /// Set on the node reached through `END`: the command and its words.
    command: Option<(Command, String)>,
}

/// The command phrases as a trie of token ids.
#[derive(Debug)]
pub struct Grammar {
    nodes: Vec<Node>,
    phrases: usize,
}

impl Grammar {
    /// Parse `commands.txt`: `ACTION<TAB>LABEL<TAB>PHRASE<TAB>IDS` per
    /// line, where ACTION is `track` or `cancel`, LABEL a COCO name or `-`,
    /// PHRASE the words, and IDS their token ids without the start and end
    /// tokens.
    pub fn parse(text: &str) -> Result<Self, String> {
        let mut grammar = Self {
            nodes: vec![Node::default()],
            phrases: 0,
        };
        for (number, line) in text
            .lines()
            .enumerate()
            .filter(|(_, l)| !l.trim().is_empty())
        {
            let fail = |what: &str| format!("commands line {}: {what}: {line:?}", number + 1);
            let mut fields = line.split('\t');
            let (Some(action), Some(label), Some(phrase), Some(ids)) =
                (fields.next(), fields.next(), fields.next(), fields.next())
            else {
                return Err(fail("want four tab-separated fields"));
            };
            let command = match action {
                "track" => {
                    Command::Track(Label::from_name(label).ok_or_else(|| fail("unknown label"))?)
                }
                "cancel" => Command::Cancel,
                _ => return Err(fail("unknown action")),
            };
            let ids = ids
                .split_whitespace()
                .map(|id| id.parse::<u32>())
                .collect::<Result<Vec<_>, _>>()
                .map_err(|_| fail("bad token id"))?;
            if ids.is_empty() {
                return Err(fail("no tokens"));
            }
            grammar.insert(&ids, command, phrase);
        }
        if grammar.phrases == 0 {
            return Err("commands: no phrases".into());
        }
        Ok(grammar)
    }

    fn insert(&mut self, ids: &[u32], command: Command, phrase: &str) {
        let mut node = 0;
        for &token in ids.iter().chain(&[END]) {
            node = match self.nodes[node].children.iter().find(|&&(t, _)| t == token) {
                Some(&(_, next)) => next,
                None => {
                    self.nodes.push(Node::default());
                    let next = self.nodes.len() - 1;
                    self.nodes[node].children.push((token, next));
                    next
                }
            };
        }
        // The first phrase to claim a token sequence keeps it.
        if self.nodes[node].command.is_none() {
            self.nodes[node].command = Some((command, phrase.to_string()));
            self.phrases += 1;
        }
    }

    pub fn phrases(&self) -> usize {
        self.phrases
    }

    /// The longest phrase in tokens, with its start and end.
    pub fn longest(&self) -> usize {
        fn depth(nodes: &[Node], node: usize) -> usize {
            nodes[node]
                .children
                .iter()
                .map(|&(_, next)| 1 + depth(nodes, next))
                .max()
                .unwrap_or(0)
        }
        1 + depth(&self.nodes, 0)
    }
}

/// A command heard, and how sure the model was.
#[derive(Clone, Debug, PartialEq)]
pub struct Recognition {
    pub command: Command,
    /// The words heard, as the grammar spells them.
    pub phrase: String,
    /// The phrase's tokens, start and end included.
    pub tokens: Vec<u32>,
    /// Mean log-probability per token of the phrase under the model's free
    /// distribution: near 0 when the model heard the phrase anyway, well
    /// below for speech that only the constraint turned into a command.
    pub confidence: f32,
}

/// Greedy decoding through `grammar`. `logits` gives the model's next-token
/// logits after a transcript (start token first).
pub fn decode(
    grammar: &Grammar,
    mut logits: impl FnMut(&[u32]) -> Vec<f32>,
) -> Option<Recognition> {
    let mut tokens = vec![START];
    let mut node = 0;
    let mut total = 0.0;
    loop {
        let children = &grammar.nodes[node].children;
        if children.is_empty() {
            break;
        }
        let scores = logits(&tokens);
        let (token, next) = *children
            .iter()
            .filter(|&&(t, _)| (t as usize) < scores.len())
            .max_by(|a, b| scores[a.0 as usize].total_cmp(&scores[b.0 as usize]))?;
        total += log_softmax(&scores, token as usize);
        tokens.push(token);
        node = next;
    }
    let steps = (tokens.len() - 1) as f32;
    let (command, phrase) = grammar.nodes[node].command.clone()?;
    Some(Recognition {
        command,
        phrase,
        tokens,
        confidence: total / steps,
    })
}

/// `log(softmax(scores)[index])`, stable for large logits.
fn log_softmax(scores: &[f32], index: usize) -> f32 {
    let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum: f32 = scores.iter().map(|&s| (s - max).exp()).sum();
    scores[index] - max - sum.ln()
}

#[cfg(test)]
mod tests {
    use super::*;

    const COMMANDS: &str = "track\tbicycle\tTrack the bicycle\t10 11 12\n\
                            track\tbicycle\tTrack the bike\t10 11 13\n\
                            track\tperson\tTrack the person\t10 11 14 15\n\
                            track\tperson\tFind the person\t20 11 14 15\n\
                            cancel\t-\tCancel\t30\n";

    /// A model that strongly expects `said` and gives everything else a
    /// flat low score.
    fn model(said: &'static [u32]) -> impl FnMut(&[u32]) -> Vec<f32> {
        move |tokens| {
            let mut scores = vec![0.0; 64];
            let next = said.get(tokens.len() - 1).copied().unwrap_or(END);
            scores[next as usize] = 12.0;
            scores
        }
    }

    #[test]
    fn grammar_parses_and_shares_prefixes() {
        let grammar = Grammar::parse(COMMANDS).unwrap();
        assert_eq!(grammar.phrases(), 5);
        // Start, four tokens, end.
        assert_eq!(grammar.longest(), 6);
        assert!(Grammar::parse("track\tunicorn\tTrack the unicorn\t1\n").is_err());
        assert!(Grammar::parse("dance\t-\tDance\t1\n").is_err());
        assert!(Grammar::parse("cancel\t-\t30\n").is_err());
        assert!(Grammar::parse("").is_err());
    }

    #[test]
    fn decoding_follows_the_model_inside_the_grammar() {
        let grammar = Grammar::parse(COMMANDS).unwrap();
        let heard = decode(&grammar, model(&[10, 11, 14, 15])).unwrap();
        assert_eq!(
            heard.command,
            Command::Track(Label::from_name("person").unwrap())
        );
        assert_eq!(heard.tokens, [START, 10, 11, 14, 15, END]);
        assert_eq!(heard.phrase, "Track the person");
        assert!(heard.confidence > -0.01, "{}", heard.confidence);
        let heard = decode(&grammar, model(&[30])).unwrap();
        assert_eq!(heard.command, Command::Cancel);
    }

    #[test]
    fn speech_outside_the_grammar_is_unsure() {
        let grammar = Grammar::parse(COMMANDS).unwrap();
        // The model hears 40 41 42, which no phrase starts with: decoding
        // still ends on a command, with low confidence.
        let heard = decode(&grammar, model(&[40, 41, 42])).unwrap();
        assert!(heard.confidence < -5.0, "{}", heard.confidence);
        // A near miss, the right start and a wrong ending, is in between.
        let near = decode(&grammar, model(&[10, 11, 40])).unwrap();
        assert!(near.confidence < -1.0 && near.confidence > heard.confidence);
    }
}
