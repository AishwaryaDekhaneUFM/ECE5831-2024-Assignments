#imports
import logic_gate as LG

#test cases
test_cases = [[0, 0], [0, 1], [1, 0], [1, 1]]

gates = lg.LogicGate()

#test
print('Test for AND Gate')
for test in test_cases:
    y = gates.and_gate(test[0], test[1])
    print(f"{y}=AND({test[0]}, {test[1]})")

print('Test for NAND Gate')
for test in test_cases:
    y = gates.nand_gate(test[0], test[1])
    print(f"{y}=NAND({test[0]}, {test[1]})")

print('Test for OR Gate')
for test in test_cases:
    y = gates.or_gate(test[0], test[1])
    print(f"{y}=OR({test[0]}, {test[1]})")

print('Test for NOR Gate')
for test in test_cases:
    y = gates.nor_gate(test[0], test[1])
    print(f"{y}=NOR({test[0]}, {test[1]})")

print('Test for XOR Gate')
for test in test_cases:
    y = gates.xor_gate(test[0], test[1])
    print(f"{y}=XOR({test[0]}, {test[1]})")
