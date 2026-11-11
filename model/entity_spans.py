"""Entity markers and masks, including nested/crossing/identical spans."""


def encode_marked_sentence(words, head_span, tail_span, tokenizer, max_length, sample_index):
    spans = {'head': tuple(head_span), 'tail': tuple(tail_span)}
    for name, (start, end) in spans.items():
        if not 0 <= start < end <= len(words):
            raise ValueError(f'Sample {sample_index}: invalid {name} span {(start, end)}')
    marker_ids = {'head': tokenizer.convert_tokens_to_ids('#'),
                  'tail': tokenizer.convert_tokens_to_ids('@')}
    rank = {'head': 0, 'tail': 1}
    boundaries = sorted({0, len(words), *(p for span in spans.values() for p in span)})
    content, openings, closings = [], {}, {}
    entity_tokens = {'head': [], 'tail': []}
    for i, boundary in enumerate(boundaries):
        # Close before opening at adjacent spans; open outer spans first.
        ending = sorted((name for name, span in spans.items() if span[1] == boundary),
                        key=lambda name: (-spans[name][0], -rank[name]))
        for name in ending:
            closings[name] = len(content)
            content.append(marker_ids[name])
        starting = sorted((name for name, span in spans.items() if span[0] == boundary),
                          key=lambda name: (-spans[name][1], rank[name]))
        for name in starting:
            openings[name] = len(content)
            content.append(marker_ids[name])
        if i + 1 == len(boundaries):
            break
        end = boundaries[i + 1]
        pieces = tokenizer.encode(' '.join(words[boundary:end]), add_special_tokens=False)
        indices = list(range(len(content), len(content) + len(pieces)))
        for name, (entity_start, entity_end) in spans.items():
            if entity_start <= boundary and end <= entity_end:
                entity_tokens[name].extend(indices)
        content.extend(pieces)

    # Include every entity marker and lexical token in the retained window.
    budget = max_length - 2
    first, last = min(openings.values()), max(closings.values())
    width = last - first + 1
    if width > budget:
        raise ValueError(f'Sample {sample_index}: entity span exceeds max_length={max_length}')
    offset = 0
    if len(content) > budget:
        offset = max(0, min(first - (budget - width) // 2, len(content) - budget))
    ids = [tokenizer.cls_token_id] + content[offset:offset + budget] + [tokenizer.sep_token_id]
    markers = tuple(openings[name] - offset + 1 for name in ['head', 'tail'])
    masks = {name: [p - offset + 1 for p in positions] for name, positions in entity_tokens.items()}
    # Shared lexical tokens may belong to both masks. Inserted markers belong to neither.
    return ids, markers, masks['head'], masks['tail']
