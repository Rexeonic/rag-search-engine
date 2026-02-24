from math import isclose

from semantic_search import dot

TestCase = tuple[list[float], list[float], float | str]

run_cases: list[TestCase] = [
    ([1, 2, 3], [1, 2, 3], 14.0),  # identical vectors
    ([1, 0, 0], [0, 1, 0], 0.0),  # orthogonal
    ([1, -2, 3], [-1, 2, -3], -14.0),  # opposite direction
    ([1, 2], [10, 20], 50.0),  # k * b scales the dot
    ([1, -1, 0], [1, 1, 0], 0.0),  # Mixed signs / cancels to zero
    ([0.5, 0.25], [0.2, 0.4], 0.2),  # Fractions / decimals
    ([1e9, 2e9], [3e9, -4e9], -5e18),  # Large magnitudes
    ([1, 2, 3], [1, 2], "exception"),  # Error: length mismatch
]

submit_cases: list[TestCase] = run_cases + [
    ([1e-9, 2e-9], [3e-9, 4e-9], 1.1e-17),  # Very small magnitudes
    ([], [], 0.0),  # Empty vectors (valid, length match)
    ([-7.0], [5.0], -35.0),  # Single-element vectors
    ([2, 3, 4], [5, 6, 7], 56.0),  # Typical mixed integers
    ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 0.0),  # All zeros
    ([1, 2, 3, 4, 5], [5, 4, 3, 2, 1], 35.0),  # Longer vector
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
        dot_product = dot(vec1, vec2)
        print(f"Actual result:   {dot_product:.2f}")
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

    if isinstance(expected, float) and isclose(dot_product, expected):
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