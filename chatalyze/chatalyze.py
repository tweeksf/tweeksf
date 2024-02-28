"""
Author: Dave Mitchell
Date: 2024-02-27
Description: Analyzes and breaks down chat logs into entities.

This script analyzes chat logs and breaks them down into people, places, gpe's and other entities. 
"""

import spacy
from spacy.pipeline import EntityRuler
import argparse
import re
from collections import defaultdict

def main():
    parser = argparse.ArgumentParser(description='Perform NER on a text file.')
    parser.add_argument('filename', type=str, help='The name of the text file to analyze.')
    args = parser.parse_args()

    with open(args.filename, 'r') as file:
        lines = file.readlines()

    nlp = spacy.load('en_core_web_sm')

    nlp.add_pipe('entity_ruler')

    ruler = nlp.get_pipe('entity_ruler')

    patterns = [{"label": "LABEL", "pattern": "pattern1"},
                {"label": "LABEL", "pattern": "pattern2"}]
    ruler.add_patterns(patterns)
    entities = defaultdict(dict)

    for line in lines:
        match = re.match(r'^(?:(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\s+)?(\w+)\s+(\w+)\s+(.*)$', line)
        if match:
            _, _, _, message = match.groups()

            doc = nlp(message)

            for entity in doc.ents:
                if '[' in entity.text or ']' in entity.text:
                    continue
                
                start = max(0, entity.start_char - 25)
                end = min(len(doc.text), entity.end_char + 25)
                snippet = doc.text[start:end]
                snippet = re.sub(r'\[.*?\]', '', snippet)
                entities[entity.label_][entity.text] = snippet

    for label, texts in entities.items():
        print(f'\n\n{label}:')
        for text, snippet in sorted(texts.items()):
            print(f'{text[:25]:<30} - "{snippet}"')
        print()

if __name__ == '__main__':
    main()