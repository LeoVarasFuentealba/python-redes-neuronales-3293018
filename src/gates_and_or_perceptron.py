from perceptron_monolayer import Perceptron
from activation import unit_step

perceptron = Perceptron(inputs=2, activation_function=unit_step)
perceptron.set_weigths([2,2])
perceptron.set_bias(-3)

print("AND")
print(f"0 0 = {perceptron.run([0,0]):.5f}")
print(f"0 1 = {perceptron.run([0,1]):.5f}")
print(f"1 0 = {perceptron.run([1,0]):.5f}")
print(f"1 1 = {perceptron.run([1,1]):.5f}")

perceptron = Perceptron(inputs=2, activation_function=unit_step)
perceptron.set_weigths([3,3])
perceptron.set_bias(-1)

print("OR")
print(f"0 0 = {perceptron.run([0,0]):.5f}")
print(f"0 1 = {perceptron.run([0,1]):.5f}")
print(f"1 0 = {perceptron.run([1,0]):.5f}")
print(f"1 1 = {perceptron.run([1,1]):.5f}")
