from perceptron_multilayer import MultilayerPerceptron

mlp = MultilayerPerceptron(layers=[2,2,1], learning_rate=0.25)
for i in range(3000):
    mse = 0.0
    mse += mlp.backprop([0,0], [0])
    mse += mlp.backprop([0,1], [1])
    mse += mlp.backprop([1,0], [1])
    mse += mlp.backprop([1,1], [0])
    mse = mse / 4

    if i % 100 == 0:
        print(mse)

mlp.print_network()

print(f"0 0 = {mlp.run([0, 0])[0]:.5f}")
print(f"0 0 = {mlp.run([0, 1])[0]:.5f}")
print(f"0 0 = {mlp.run([1, 0])[0]:.5f}")
print(f"0 0 = {mlp.run([1, 1])[0]:.5f}")


