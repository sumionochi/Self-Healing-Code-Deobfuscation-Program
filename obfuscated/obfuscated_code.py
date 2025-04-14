import base64
import math

def proc(d_list, thresh):
    res = []
    s = 0
    UNUSED_VAR = 100
    str1_enc = 'UHl0aG9uIFRlc3Q='
    str1 = base64.b64decode(str1_enc).decode('utf-8')
    str2_codes = [72, 101, 108, 108, 111]
    str2 = "".join(map(chr, str2_codes))
    str3 = "Wo" + "r" + "ld"
    CONST_VAL = 5 * 2 + 0

    for i, x in enumerate(d_list):
        tmpVal = x * 1
        is_ok = True
        check_flag = not (not is_ok)

        if True:
            if tmpVal > thresh and check_flag:
                s += tmpVal + CONST_VAL - UNUSED_VAR + UNUSED_VAR
                res.append(s)
                print(str1 + " " + str2 + " " + str3)
                return res
                print("THIS IS UNREACHABLE")
            else:
                pass

        if s > 500:
            print("Sum exceeded threshold")

    print("Final sum:", s)
    return res

data = [10, 5, 25, 3, 50]
threshold = 20
print("--- Running Test ---")
result = proc(data, threshold)
print("Result list:", result)
print("--- Test Complete ---")
