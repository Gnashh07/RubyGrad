######### TEST1 ##################

# Create an MLP with 2 inputs, a hidden layer of 3 neurons, and 1 output
mlp = MLP.new(2, [3, 1])

# Use the MLP
x = [Value.new(1.0, 'x1'), Value.new(-1.0, 'x2')]
output = mlp.call(x)

# Backward pass
output.backward if output.is_a?(Value)
output.each(&:backward) if output.is_a?(Array)

# Visualize the computation graph
dot = draw_dot(output.is_a?(Array) ? output.last : output)
dot.output(svg: 'mlp_graph.svg')

################# TEST2 #######################

# Define your input data
xs = [
  [2.0, 3.0, -1.0],
  [3.0, -1.0, 0.5],
  [0.5, 1.0, 1.0],
  [1.0, 1.0, -1.0],
]

# Define your target outputs
ys = [1.0, -1.0, -1.0, 1.0]

# Create your MLP
# Note: We're using 3 input neurons to match your xs data, and 1 output neuron to match ys
n = MLP.new(3, [4, 4, 1])

# Training loop
20.times do |k|
  # Forward pass
  ypred = xs.map { |x| n.call(x.map { |val| Value.new(val) }) }
  loss = ypred.zip(ys).map { |yout, ygt| (yout - Value.new(ygt))**2 }.reduce(:+)


  loss_value = loss.data  # Get the numeric value from the loss for summation

  # Backward pass
  n.parameters.each { |p| p.grad = 0.0 }
  loss.backward

  # Update
  n.parameters.each { |p| p.data += -0.1 * p.grad }

  puts "#{k} #{loss_value}"
end

# Test the trained network
puts "\nFinal predictions:"
xs.each_with_index do |x, i|
  pred = n.call(x.map { |val| Value.new(val) })
  puts "Input: #{x}, Predicted: #{pred.data.round(4)}, Target: #{ys[i]}"
end