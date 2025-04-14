def dead_code_example(x):
    # Case 1: Early return makes subsequent code unreachable.
    if x > 10:
        result = x * 2
        print("This message will never be printed because of the return below.")
        return result  # After return, any code is unreachable.
        print("Dead code: This print is unreachable.")
    
    # Case 2: An if condition that is always false.
    if False:
        result = x + 100  # This branch is never executed.
    else:
        result = x - 5

    # Case 3: Unnecessary 'if True' wrapping that adds no value.
    if True:
        temp = result + 10
        # Return is called immediately so the next line is never executed.
        return temp
        print("This print is dead due to the return above.")
    else:
        result = result * 2  # This else branch is never executed because condition is always True.

    # Case 4: Code after a return in a loop – unreachable code.
    for i in range(5):
        print("Loop iteration", i)
        return result
        print("Dead code inside loop: This will never print.")

    # Code outside loop that would never execute if any loop iteration returns.
    print("This code is unreachable if the loop returns early.")
    return result

# Test the function:
print("Result when x=15:", dead_code_example(15))
print("Result when x=3:", dead_code_example(3))
