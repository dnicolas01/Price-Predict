
#while True:
    #predict = input("What's the prediction price: ")
    #actual = input("What's the actual price: ")
    #accuracy = abs((abs((float(predict)-float(actual)))/float(actual) * 100)-100)
    #print(f"your accuracy is {accuracy} %")
    #answer = input("would you like to go again? (y/n): ").lower()
    #if answer == 'y':
        #continue
    #else:
        #break

num1 = input("Enter your value: ")
num1 = float(num1)
num2 = input("Enter your value: ")
num2 = float(num2)
num3 = input("Enter your value: ")
num3 = float(num3)
num4 = input("Enter your value: ")
num4 = float(num4)
num5 = input("Enter your value: ")
num5 = float(num5)
num = num1+num2+num3+num4+num5
average = num/5
print(f' Your average is: {average}')
