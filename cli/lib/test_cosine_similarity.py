from math import isclose

from semantic_search import cosine_similarity

TestCase = tuple[list[float], list[float], float | str]

run_cases: list[TestCase] = [
    ([1, 2, 3], [1, 2, 3], 1.0),  # Identical vectors → 1.0
    ([1, 0, 0], [0, 1, 0], 0.0),  # Orthogonal vectors → 0.0
    ([1, 0], [-1, 0], -1.0),  # Opposite direction → -1.0
    ([1, 2], [10, 20], 1.0),  # Scaling invariance → 1.0
    ([1, -1, 0], [1, 1, 0], 0.0),  # Mixed signs but orthogonal → 0.0
    ([1.0, 0.0], [0.5, 3**0.5 / 2], 0.5),  # Known 60° angle → cos = 0.5
    ([1, 2], [1, 2, 3], "exception"),  # Mismatched lengths → exception
]

submit_cases: list[TestCase] = run_cases + [
    ([0, 0, 0], [1, 2, 3], 0.0),  # Zero vector present → 0.0
    ([0.1, 0.2], [0.3, 0.6], 1.0),  # Float proportional vectors → 1.0
    ([1.0, 1e-12], [0.0, 1.0], 0.0),  # Nearly orthogonal with tiny component
    ([1.0, 0.0], [-0.5, 3**0.5 / 2], -0.5),  # 120° angle → cos = -0.5
]


def test(vec1: list[float], vec2: list[float], expected: float | str) -> bool:
    print("---------------------------------")
    print(f"Vector 1: {vec1}")
    print(f"Vector 2: {vec2}")

    print(
        f"\nExpected result: {expected:.2f}"
        if isinstance(expected, float)
        else f"\nExpected result: {expected}"
    )

    try:
        similarity = cosine_similarity(vec1, vec2)
        print(f"Actual result:   {similarity:.2f}")
    except ValueError:
        print("Actual result:   exception")
        if expected == "exception":
            print("\nPass")
            return True
        else:
            print("\nFail")
            return False
    except Exception as e:
        print(f"Actual result:   {e}")
        print("\nFail")
        return False

    if isinstance(expected, float) and isclose(similarity, expected, abs_tol=1e-12):
        print("\nPass")
        return True

    print("\nFail")
    return False


def main() -> None:
    passed = 0
    failed = 0
    skipped = len(submit_cases) - len(test_cases)

    for test_case in test_cases:
        correct = test(*test_case)
        if correct:
            passed += 1
        else:
            failed += 1

    if failed == 0:
        print("============= PASS ==============")
    else:
        print("============= FAIL ==============")

    if skipped > 0:
        print(f"{passed} passed, {failed} failed, {skipped} skipped")
    else:
        print(f"{passed} passed, {failed} failed")


test_cases: list[TestCase] = submit_cases
if "__RUN__" in globals():
    test_cases = run_cases

main()
