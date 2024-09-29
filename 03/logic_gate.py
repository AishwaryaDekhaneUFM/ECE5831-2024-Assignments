#imports
import numpy as np


class LogicGate():
    def __init__(self) -> None:
        pass

    def and_gate(self, x1, x2) -> int:
        b = -0.7
        w = np.array([0.5, 0.5, 1])
        x = np.array([x1, x2, b])

        y = np.sum(x*w)

        if y > 0:
            return 1
        else:
            return 0

    def nand_gate(self, x1, x2) -> int:
        b = 0.7
        w = np.array([-0.5, -0.5, 1])
        x = np.array([x1, x2, b])

        y = np.sum(x*w)

        if y > 0:
            return 1
        else:
            return 0
    
    def or_gate(self, x1, x2) -> int:
        b = -0.9
        w = np.array([1, 1, 1])
        x = np.array([x1, x2, b])

        y = np.sum(x*w)

        if y > 0:
            return 1
        else:
            return 0
    
    def nor_gate(self, x1, x2) -> int:
        b = 0.9
        w = np.array([-1, -1, 1])
        x = np.array([x1, x2, b])

        y = np.sum(x*w)

        if y > 0:
            return 1
        else:
            return 0
    
    def xor_gate(self, x1, x2) -> int:
        y1 = self.or_gate(x1, x2)
        y2 = self.nand_gate(x1, x2)
        return self.and_gate(y1, y2)

# Help message logic
if __name__ == "__main__":
    help_message = """
    The `LogicGate` class provides a simplified simulation of basic digital logic gates (AND, NAND, OR, NOR, and XOR) using linear combinations and threshold comparisons. The class does not require any initialization parameters, as indicated by the placeholder `__init__` method. Each logic gate method takes two binary inputs, `x1` and `x2`, and uses predefined weights and bias values to simulate the gate's behavior. The weights and biases are combined with the inputs and processed through a weighted sum calculation, `y`. The result is then compared to a threshold (usually zero), and the method returns `1` if the threshold is surpassed, or `0` otherwise.

For instance, the `and_gate` method uses weights of `0.5` for each input and a bias of `-0.7`. In contrast, the `nand_gate` method negates these weights and adjusts the bias to `0.7` to invert the output. The `or_gate` method uses weights of `1` and a bias of `-0.9` to ensure that any non-zero input produces an output of `1`, while the `nor_gate` reverses both weights and bias. The `xor_gate` method is unique in that it uses a combination of the other gates: it first evaluates the `or_gate` and `nand_gate` methods and then combines their results using the `and_gate` to produce the final exclusive-or behavior. This class demonstrates using simple linear algebra to simulate the fundamental logic operations at the core of digital systems.
    """
    print(help_message)
