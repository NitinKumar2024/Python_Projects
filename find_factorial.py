
def find_factorial(number):

    if number == 0 or number == 1:
        return 1
    else:
        return number * find_factorial(number - 1)

while True:
    number = int(input("Enter Your Number: "))

    if number == 00:
        print(1)
        break
    else:
        print(find_factorial(number))
