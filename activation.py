import sympy as sp


class Activation:
    def __init__(self, x, y):
        """ function represents a sympy expression with function[0] as the variable
        and function[1] as the function."""
        # self.function = function    # activation function
        self.x = sp.Symbol(x)
        self.y = sp.sympify(y)
        self.f = sp.lambdify(x, y, 'numpy')
        self.f_prime = sp.lambdify(x, sp.diff(y, x), 'numpy')

    def activate(self, x):
        return self.f(x)

    def derivative(self, x):
        return self.f_prime(x)

    def __call__(self, *args, **kwargs):
        return self.activate(*args, **kwargs)

    def D(self, x):
        return self.derivative(x)


sigmoid = Activation('x', '1 / (1 + exp(-x))')
tanh = Activation('x', '2 / (1 + exp(-2 * x)) - 1')
swish = Activation('x', 'x / (1 + exp(-x))')
# leaky_relu = Activation('x', 'x if x > 0 else 0.01 * x')

