import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'model'))
from entity_spans import encode_marked_sentence


class WordTokenizer:
    """Test double for span bookkeeping; not a replacement for BERT tokenization."""
    cls_token_id, sep_token_id = 1, 2

    def __init__(self):
        self.vocab = {'#': 3, '@': 4}

    def convert_tokens_to_ids(self, token):
        if token not in self.vocab:
            self.vocab[token] = len(self.vocab) + 3
        return self.vocab[token]

    def encode(self, text, add_special_tokens=False):
        return [self.convert_tokens_to_ids(token) for token in text.split()]


class EntitySpanTest(unittest.TestCase):
    def check_spans(self, words, head, tail, max_length=512):
        tokenizer = WordTokenizer()
        ids, markers, h, t = encode_marked_sentence(words, head, tail, tokenizer, max_length, 23)
        self.assertEqual([ids[p] for p in h], tokenizer.encode(' '.join(words[head[0]:head[1]])))
        self.assertEqual([ids[p] for p in t], tokenizer.encode(' '.join(words[tail[0]:tail[1]])))
        self.assertEqual([ids[p] for p in markers], [3, 4])
        self.assertFalse(set(markers) & (set(h) | set(t)))
        self.assertLessEqual(len(ids), max_length)
        return ids, h, t

    def test_nested_crossing_identical_adjacent_and_reverse(self):
        words = ['Virginia', 'House', 'of', 'Delegates']
        for head, tail in [([0, 4], [0, 1]), ([0, 1], [0, 4]),
                           ([0, 3], [1, 4]), ([0, 4], [0, 4]),
                           ([0, 2], [2, 4]), ([2, 4], [0, 2])]:
            with self.subTest(head=head, tail=tail):
                ids, h, t = self.check_spans(words, head, tail)
                self.assertEqual(len(ids), len(words) + 6)  # no duplicated source tokens
                self.assertEqual(len(set(h) & set(t)), max(0, min(head[1], tail[1]) - max(head[0], tail[0])))

    def test_long_context_and_literal_markers(self):
        words = ['#'] + ['context'] * 40 + ['Virginia', '@', 'Delegates'] + ['context'] * 40
        self.check_spans(words, [41, 44], [41, 42], max_length=32)

    def test_disjoint_token_stream_unchanged(self):
        tokenizer = WordTokenizer()
        words = ['alice', 'met', 'bob']
        ids, _, _, _ = encode_marked_sentence(words, [0, 1], [2, 3], tokenizer, 512, 0)
        expected = [1] + tokenizer.encode('# alice # met @ bob @') + [2]
        self.assertEqual(ids, expected)

    def test_invalid_and_too_distant(self):
        tokenizer = WordTokenizer()
        with self.assertRaises(ValueError):
            encode_marked_sentence(['a'], [0, 0], [0, 1], tokenizer, 512, 0)
        with self.assertRaises(ValueError):
            encode_marked_sentence(['a'] * 100, [0, 1], [99, 100], tokenizer, 32, 0)

    def test_all_local_wikizsl_spans(self):
        path = ROOT / 'data/wikizsl/wikizsl_dataset.json'
        if not path.is_file():
            self.skipTest('Full Wiki-ZSL data not available')
        raw = json.loads(path.read_text(encoding='utf-8'))
        overlap = 0
        for idx, item in enumerate(raw):
            h, t = item['h']['pos'], item['t']['pos']
            try:
                self.check_spans(item['token'], h, t)
            except Exception as exc:
                raise AssertionError(f'Wiki-ZSL sample {idx}') from exc
            overlap += not (h[1] <= t[0] or t[1] <= h[0])
        print(f'Validated {len(raw)} Wiki-ZSL samples including {overlap} overlaps (word-tokenizer test double).')


if __name__ == '__main__':
    unittest.main()
