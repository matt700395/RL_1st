# 2. Estimating PI using Circle and Square
# Run the following code in local machine to visualize the animation and get the output.
# Import required libraries :
import turtle
import random
import matplotlib.pyplot as plt
import math

# To visualize the random points :
myPen = turtle.Turtle()
myPen.hideturtle()
myPen.speed(0)

# Drawing a square :
myPen.up()
myPen.setposition(-100, -100)
myPen.down()
myPen.fd(200)
myPen.left(90)
myPen.fd(200)

myPen.left(90)
myPen.fd(200)
myPen.left(90)
myPen.fd(200)
myPen.left(90)

# Drawing a circle :
myPen.up()
myPen.setposition(0, -100)
myPen.down()
myPen.circle(100) #r=100

# To count the points inside and outside the circle :
in_circle = 0
out_circle = 0

# To store the values of PI :
pi_values = []
out_of_pi_values = []

# Running for 5 times :
for i in range(5):
    for j in range(1000):
    #for j in range(10):

        # Generate random numbers :
        x = random.randrange(-100, 100)
        y = random.randrange(-100, 100)

        # Check if the number lies outside the circle :
        if (x ** 2 + y ** 2 > 100 ** 2): #out fo circle
            myPen.color("black")
            myPen.up()
            myPen.goto(x, y)
            myPen.down()
            myPen.dot()
            out_circle = out_circle + 1

        else: # in circle
            myPen.color("red")
            myPen.up()
            myPen.goto(x, y)
            myPen.down()
            myPen.dot()
            in_circle = in_circle + 1

        # Calculating the value of PI :
        pi = 4.0 * in_circle / (in_circle + out_circle)
        out_of_pi = 4.0 * out_circle / (in_circle + out_circle)

        # Append the values of PI in list :
        pi_values.append(pi)
        out_of_pi_values.append(out_of_pi)

        # Calculating the errors :
        avg_pi_errors = [abs(math.pi - pi) for pi in pi_values]

    # Print the final value of PI for each iterations :
    #print(pi_values[-1])
    #print(pi_values)
    print(out_of_pi_values[-1])

print(f'final value : {out_of_pi_values[-1]} ')


# Plot the PI values :
plt.axhline(y=math.pi, color='g', linestyle='-')
plt.plot(out_of_pi_values)
plt.xlabel("Iterations")
plt.ylabel("Value of PI")
plt.show()

# Plot the error in calculation :
plt.axhline(y=0.0, color='g', linestyle='-')
plt.plot(avg_pi_errors)
plt.xlabel("Iterations")
plt.ylabel("Error")
plt.show()








