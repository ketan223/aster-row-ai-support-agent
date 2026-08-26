import os
import sys
import traceback

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.test_behavioral import get_all_test_cases, test_behavioral_case

# Runs all evaluation test cases (visible and custom), prints individual results,
# and outputs category statistics and overall pass rates.
def run_evaluation():
    print("==================================================")
    print("ASTER & ROW SUPPORT AGENT - DETERMINISTIC EVALUATION")
    print("==================================================")
    
    cases = get_all_test_cases()
    print(f"Total Cases Found: {len(cases)}\n")
    
    results = []
    category_stats = {} # {category: {"passed": 0, "total": 0}}
    
    for case in cases:
        case_id = case["id"]
        category = case.get("category", "unspecified")
        
        # Initialize category stats
        if category not in category_stats:
            category_stats[category] = {"passed": 0, "total": 0}
        category_stats[category]["total"] += 1
        
        print(f"Running {case_id} [{category}]...", end="", flush=True)
        try:
            test_behavioral_case(case)
            print(" PASS")
            results.append((case_id, category, True, None))
            category_stats[category]["passed"] += 1
        except AssertionError as ae:
            print(" FAIL")
            print(f"  Reason: {str(ae)}")
            results.append((case_id, category, False, str(ae)))
        except Exception as e:
            print(" ERROR")
            print(f"  Error: {str(e)}")
            traceback.print_exc()
            results.append((case_id, category, False, str(e)))
            
    print("\n==================================================")
    print("CATEGORY BREAKDOWN:")
    print("==================================================")
    
    passed_total = 0
    grand_total = 0
    
    for cat, stats in sorted(category_stats.items()):
        p = stats["passed"]
        t = stats["total"]
        pct = (p / t * 100) if t > 0 else 0
        print(f"{cat.capitalize():<25} : {p}/{t} passed ({pct:.1f}%)")
        passed_total += p
        grand_total += t
        
    overall_pct = (passed_total / grand_total * 100) if grand_total > 0 else 0
    print("==================================================")
    print(f"Overall: {passed_total}/{grand_total}")
    print(f"Pass rate: {overall_pct:.1f}%")
    print("==================================================")
    
    # Exit with code 1 if any tests failed
    if passed_total < grand_total:
        sys.exit(1)
    else:
        sys.exit(0)

if __name__ == "__main__":
    run_evaluation()
