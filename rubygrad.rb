require 'graphviz'
require 'set'

class Value
  attr_accessor :data, :label, :grad, :_backward, :_op, :prev

  def initialize(data, label = "", _op = "", prev = [])
    @data = data
    @grad = 0.0
    @_backward = nil
    @_op = _op.to_s
    @label = label
    @prev = prev.is_a?(Array) ? prev : [prev]
    puts "Initialized Value: data=#{@data}, label=#{@label}, children=#{@prev.map { |v| v.is_a?(Value) ? v.label : "N/A" }}"
  end

  def to_s
    "Value(data=#{@data})"
  end

  def +(other)
    other = other.is_a?(Value) ? other : Value.new(other)
    out = Value.new(@data + other.data, '', '+', [self, other])

    out.instance_variable_set(:@_backward, -> {
      @grad += 1.0 * out.grad
      other.grad += 1.0 * out.grad
    })

    out
  end

  def *(other)
    other = other.is_a?(Value) ? other : Value.new(other)
    out = Value.new(@data * other.data, '', '*', [self, other])

    out.instance_variable_set(:@_backward, -> {
      @grad += other.data * out.grad
      other.grad += @data * out.grad
    })

    out
  end

  def **(other)
    raise "only supporting int/float powers for now" unless other.is_a?(Integer) || other.is_a?(Float)
    out = Value.new(@data**other, '', "**#{other}", [self])

    out.instance_variable_set(:@_backward, -> {
      @grad += other * (@data**(other - 1)) * out.grad
    })

    out
  end

  def /(other)
    self * (other**-1)
  end

  def -(other)
    self + (-other)
  end

  def tanh
    x = @data
    t = (Math.exp(2 * x) - 1) / (Math.exp(2 * x) + 1)
    out = Value.new(t, '', 'tanh', [self])

    out.instance_variable_set(:@_backward, -> {
      @grad += (1 - t**2) * out.grad
    })

    out
  end

  def exp
    x = @data
    out = Value.new(Math.exp(x), '', 'exp', [self])

    out.instance_variable_set(:@_backward, -> {
      @grad += out.data * out.grad
    })

    out
  end

  def backward
    topo = []
    visited = Set.new

    build_topo(self, visited, topo)
    @grad = 1.0

    topo.reverse_each do |node|
      node.instance_variable_get(:@_backward)&.call
    end
  end

  private

  def build_topo(v, visited = Set.new, topo = [])
    return if visited.include?(v)

    v.prev.each do |child|
      build_topo(child, visited, topo)
    end

    visited.add(v)
    topo << v
  end
end

def trace(root)
  nodes, edges = Set.new, []
  stack = [root]

  until stack.empty?
    v = stack.pop
    next if nodes.include?(v)

    nodes.add(v)
    v.prev.each do |child|
      edges << [child, v]
      stack.push(child)
    end
  end

  [nodes.to_a, edges]
end

def draw_dot(root)
  dot = GraphViz.new(:G, type: :digraph, rankdir: 'LR')

  nodes, edges = trace(root)

  nodes.each do |n|
    label = n.label.to_s.empty? ? 'unnamed' : n.label.to_s.gsub(/[^a-zA-Z0-9_]/, '')
    uid = n.object_id.to_s
    op = n._op.to_s

    dot.add_node(uid, label: "{ #{label} | data #{n.data.round(4)} | grad #{n.grad.round(4)} }", shape: 'record')

    unless op.empty?
      op_uid = uid + op
      dot.add_node(op_uid, label: op)
      dot.add_edge(op_uid, uid)
    end
  end

  edges.each do |n1, n2|
    n2_op = n2._op.to_s
    dot.add_edge(n1.object_id.to_s, n2.object_id.to_s + n2_op)
  end

  dot
end

class Neuron
  attr_reader :w, :b

  def initialize(nin)
    @w = Array.new(nin) { Value.new(rand * 2 - 1, "w#{_1}") }
    @b = Value.new(rand * 2 - 1, 'b')
  end

  def call(x)
    act = @w.zip(x).map { |wi, xi| wi * xi }.reduce(:+) + @b
    out = act.tanh
    out.label = 'out'
    out
  end

  def parameters
    @w + [@b]
  end
end

class Layer
  attr_reader :neurons

  def initialize(nin, nout)
    @neurons = Array.new(nout) { Neuron.new(nin) }
  end

  def call(x)
    outs = @neurons.map.with_index { |n, i| n.call(x).tap { |o| o.label = "n#{i}_out" } }
    outs.length == 1 ? outs[0] : outs
  end

  def parameters
    @neurons.flat_map(&:parameters)
  end
end

class MLP
  attr_reader :layers

  def initialize(nin, nouts)
    sz = [nin] + nouts
    @layers = sz.each_cons(2).map.with_index do |(nin, nout), i|
      Layer.new(nin, nout).tap { |l| l.instance_variable_set(:@label, "layer#{i}") }
    end
  end

  def call(x)
    output = @layers.reduce(x) do |input, layer|
      layer.call(input)
    end
    
    if output.is_a?(Array)
      output.each_with_index { |o, i| o.label = "final_output_#{i}" }
    else
      output.label = "final_output"
    end
    
    output
  end

  def parameters
    @layers.flat_map(&:parameters)
  end
end
