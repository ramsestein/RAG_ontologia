#!/usr/bin/env python3
"""
Temporary script to analyze annotations by strategy count.
Categorizes concepts by how many strategies detected them (1, 2, or 3)
and outputs the total FP and TP for each category.
"""

import json
from pathlib import Path
from collections import defaultdict
from itertools import combinations


def get_combination_key(sources):
    """Get a canonical key for a combination of sources."""
    return tuple(sorted(sources))


def main():
    # Load the data
    input_path = Path(__file__).parent / "01_all_positives.json"
    
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Initialize counters for each category (by number of strategies)
    categories = {
        1: {"TP": 0, "FP": 0, "by_note": defaultdict(lambda: {"TP": [], "FP": []})},
        2: {"TP": 0, "FP": 0, "by_note": defaultdict(lambda: {"TP": [], "FP": []})},
        3: {"TP": 0, "FP": 0, "by_note": defaultdict(lambda: {"TP": [], "FP": []})},
    }
    
    # Track by individual strategy
    by_strategy = defaultdict(lambda: {"TP": 0, "FP": 0})
    
    # Track by strategy combination
    by_combination = defaultdict(lambda: {"TP": 0, "FP": 0})
    
    # Process all annotations
    for note in data:
        note_id = note["note_id"]
        for annotation in note["annotations"]:
            sources = annotation["source"]
            num_strategies = len(sources)
            status = annotation["status"]
            text = annotation["text"]
            
            # Count by number of strategies
            if num_strategies in categories:
                categories[num_strategies][status] += 1
                categories[num_strategies]["by_note"][note_id][status].append({
                    "text": text,
                    "source": sources
                })
            
            # Count by individual strategy (each strategy gets credit)
            for strategy in sources:
                by_strategy[strategy][status] += 1
            
            # Count by exact combination
            combo_key = get_combination_key(sources)
            by_combination[combo_key][status] += 1
    
    # Print results by number of strategies
    print("=" * 70)
    print("ANALYSIS BY NUMBER OF STRATEGIES")
    print("=" * 70)
    
    for num_strategies in [1, 2, 3]:
        cat = categories[num_strategies]
        total = cat["TP"] + cat["FP"]
        print(f"\n{'─' * 70}")
        print(f"CONCEPTS DETECTED BY {num_strategies} STRATEGY/IES")
        print(f"{'─' * 70}")
        print(f"  Total annotations: {total}")
        print(f"  True Positives (TP): {cat['TP']}")
        print(f"  False Positives (FP): {cat['FP']}")
        if total > 0:
            precision = cat['TP'] / total * 100
            print(f"  Precision: {precision:.2f}%")
        
        # Show details by note
        for note_id in sorted(cat["by_note"].keys()):
            note_data = cat["by_note"][note_id]
            tp_list = note_data["TP"]
            fp_list = note_data["FP"]
            print(f"\n  📄 Note {note_id}:")
            
            if tp_list:
                print(f"    ✅ TP ({len(tp_list)}):")
                for item in tp_list:
                    sources_str = ", ".join(item["source"])
                    print(f"       • \"{item['text']}\" [{sources_str}]")
            
            if fp_list:
                print(f"    ❌ FP ({len(fp_list)}):")
                for item in fp_list:
                    sources_str = ", ".join(item["source"])
                    print(f"       • \"{item['text']}\" [{sources_str}]")
    
    # Summary by number of strategies
    print(f"\n{'=' * 70}")
    print("SUMMARY BY NUMBER OF STRATEGIES")
    print("=" * 70)
    total_tp = sum(cat["TP"] for cat in categories.values())
    total_fp = sum(cat["FP"] for cat in categories.values())
    total = total_tp + total_fp
    print(f"Total annotations: {total}")
    print(f"Total TP: {total_tp}")
    print(f"Total FP: {total_fp}")
    if total > 0:
        print(f"Overall Precision: {total_tp / total * 100:.2f}%")
    
    # Precision by individual strategy
    print(f"\n{'=' * 70}")
    print("PRECISION BY INDIVIDUAL STRATEGY")
    print("=" * 70)
    print("(Note: annotations can be counted multiple times if detected by multiple strategies)\n")
    
    for strategy in sorted(by_strategy.keys()):
        stats = by_strategy[strategy]
        total = stats["TP"] + stats["FP"]
        precision = stats["TP"] / total * 100 if total > 0 else 0
        print(f"  {strategy}:")
        print(f"    TP: {stats['TP']}, FP: {stats['FP']}, Total: {total}")
        print(f"    Precision: {precision:.2f}%")
        print()
    
    # Precision by strategy combination
    print(f"{'=' * 70}")
    print("PRECISION BY STRATEGY COMBINATION")
    print("=" * 70)
    print("(Exact combination that detected the concept)\n")
    
    # Sort by number of strategies, then alphabetically
    sorted_combos = sorted(by_combination.keys(), key=lambda x: (len(x), x))
    
    for combo in sorted_combos:
        stats = by_combination[combo]
        total = stats["TP"] + stats["FP"]
        precision = stats["TP"] / total * 100 if total > 0 else 0
        combo_str = " + ".join(combo)
        print(f"  {combo_str}:")
        print(f"    TP: {stats['TP']}, FP: {stats['FP']}, Total: {total}")
        print(f"    Precision: {precision:.2f}%")
        print()


if __name__ == "__main__":
    main()
