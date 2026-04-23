import numpy as np


# converitng binary to decimal (unsigned)
def bin2dec(arr, n):
    dec = 0
    for i in range(n):
        dec += int(arr[i]) * (2 ** i)
    return int(dec)


# converting binary to decimal (signed 2's complement)
def bin2dec_signed(arr, n):
    # Check sign bit (MSB)
    if arr[n - 1] == 1:
        # Negative number: convert from 2's complement
        # Invert all bits and add 1, then negate
        inverted = [(1 - int(bit)) for bit in arr]
        magnitude = bin2dec(inverted, n) + 1
        return -magnitude
    else:
        # Positive number
        return bin2dec(arr, n)


# converting decimal to binary (unsigned)
def dec2bin(dec, n):
    bin_arr = [0] * n
    for i in range(n):
        bin_arr[i] = dec % 2
        dec = dec // 2
    return bin_arr


# converting decimal to binary (signed 2's complement)
def dec2bin_signed(dec, n):
    if dec >= 0:
        return dec2bin(dec, n)
    else:
        # Negative number: convert to 2's complement
        # Take absolute value, convert to binary, invert, add 1
        magnitude = -dec
        bin_mag = dec2bin(magnitude, n)
        # Invert all bits
        inverted = [(1 - bit) for bit in bin_mag]
        # Add 1
        carry = 1
        for i in range(n):
            inverted[i] += carry
            if inverted[i] > 1:
                inverted[i] = 0
                carry = 1
            else:
                carry = 0
        return inverted



# 8 instructions, 13 bits for the address
'''
000 - LOAD  mem[address] -> reg
001 - STORE reg -> mem[address]
010 - ADD   reg + mem[address] -> reg
011 - BNZ   if reg != 0, PC = address
100 - AND   reg & mem[address] -> reg
101 - NAND  ~(reg & mem[address]) -> reg
110 - XOR   reg ^ mem[address] -> reg
111 - NOT   ~reg -> reg
'''

# defining the opcodes as binary arrays
LOAD = dec2bin(0, 3)
STORE = dec2bin(1, 3)
ADD = dec2bin(2, 3)
BNZ = dec2bin(3, 3)
AND = dec2bin(4, 3)
NAND = dec2bin(5, 3)
XOR = dec2bin(6, 3)
NOT = dec2bin(7, 3)

# line numbers to store constants and varibales:
BC = 99
ONE = 100
BIT = 101
A = 102
B = 103
P = 104
ALL_ONES = 105

ZERO = 0
BACK = 1
CONT = 10



cols, rows = 16, 2**13
mem = np.zeros((rows, cols)) # make a 2D array of zeros with dimensions (rows, cols)

reg = np.zeros(cols)
pc = 1


# program starts here:
mem[ZERO] = dec2bin(0, cols) # store the constant 0 at line 0
mem[1, :] = dec2bin(0, 3) + dec2bin(ZERO, 13) # LOAD 0 (initialize P)
mem[2, :] = dec2bin(1, 3) + dec2bin(P, 13) # STORE P
mem[3, :] = dec2bin(0, 3) + dec2bin(ONE, 13) # LOAD 1 (initialize BIT)
mem[4, :] = dec2bin(1, 3) + dec2bin(BIT, 13) # STORE BIT
mem[5, :] = dec2bin(0, 3) + dec2bin(B, 13) # LOAD B
mem[6, :] = dec2bin(7, 3) + dec2bin(ZERO, 13) # NOT (BC = NOT B)
mem[7, :] = dec2bin(1, 3) + dec2bin(BC, 13) # STORE BC
mem[8, :] = dec2bin(0, 3) + dec2bin(ZERO, 13) # (unused padding)
mem[9, :] = dec2bin(0, 3) + dec2bin(ZERO, 13) # (unused padding)

# Main loop starts here at line 10
mem[10, :] = dec2bin(0, 3) + dec2bin(BC, 13) # LOAD BC
mem[11, :] = dec2bin(4, 3) + dec2bin(BIT, 13) # AND BIT (check if bit is set)
mem[12, :] = dec2bin(3, 3) + dec2bin(16, 13) # BNZ to CONT (skip if bit not set)
mem[13, :] = dec2bin(0, 3) + dec2bin(P, 13) # LOAD P
mem[14, :] = dec2bin(2, 3) + dec2bin(A, 13) # ADD A (add shifted A)
mem[15, :] = dec2bin(1, 3) + dec2bin(P, 13) # STORE P

# CONT: continue from here
mem[16, :] = dec2bin(0, 3) + dec2bin(BIT, 13) # LOAD BIT
mem[17, :] = dec2bin(2, 3) + dec2bin(BIT, 13) # ADD BIT (left shift)
mem[18, :] = dec2bin(1, 3) + dec2bin(BIT, 13) # STORE BIT
mem[19, :] = dec2bin(0, 3) + dec2bin(A, 13) # LOAD A
mem[20, :] = dec2bin(2, 3) + dec2bin(A, 13) # ADD A (left shift)
mem[21, :] = dec2bin(1, 3) + dec2bin(A, 13) # STORE A
mem[22, :] = dec2bin(3, 3) + dec2bin(10, 13) # BNZ to BACK (repeat loop)
mem[23, :] = dec2bin(0, 3) + dec2bin(ONE, 13) # LOAD 1
mem[24, :] = dec2bin(3, 3) + dec2bin(ZERO, 13) # BNZ to 0 (stop)


Avalue = 31
Bvalue = 31
mem[A] = dec2bin_signed(Avalue, cols)

print(f"\nmem[A]: {mem[A]}")
mem[B] = dec2bin_signed(Bvalue, cols)
mem[P] = dec2bin_signed(0, cols)
mem[ONE] = dec2bin(1, cols)

data = np.zeros(10)

sig_add_count = 0
sig_pp_count = 0

while pc > 0:
    instr = mem[pc]
    opcode = bin2dec(instr[0:3], 3)
    address = bin2dec(instr[3:16], 13)
    pc = (pc + 1) % rows # increment the program counter and wrap around if it exceeds the number of rows
    data [9] += 1
    if pc == 14:
        sig_add_count += 1
        sig_pp_count += 1

    match opcode:
        case 0: # LOAD
            reg = mem[address].copy()
            data[0] += 1
        case 1: # STORE
            mem[address] = reg.copy()
            data[1] += 1
        case 2: # ADD (signed 2's complement)
            reg_val = bin2dec_signed(reg, cols)
            mem_val = bin2dec_signed(mem[address], cols)
            result = reg_val + mem_val
            reg = np.array(dec2bin_signed(result, cols))
            data[2] += 1
        case 3: # BNZ
            reg_val = bin2dec(reg, cols)
            if reg_val != 0:
                pc = address
            data[3] += 1
        case 4: # AND
            reg_val = bin2dec(reg, cols)
            mem_val = bin2dec(mem[address], cols)
            result = reg_val & mem_val
            reg = np.array(dec2bin(result, cols))
            data[4] += 1
        case 5: # NAND
            reg_val = bin2dec(reg, cols)
            mem_val = bin2dec(mem[address], cols)
            result = (reg_val & mem_val) ^ ((1 << cols) - 1)  # NAND = NOT(AND)
            reg = np.array(dec2bin(result, cols))
            data[5] += 1
        case 6: # XOR
            reg_val = bin2dec(reg, cols)
            mem_val = bin2dec(mem[address], cols)
            result = reg_val ^ mem_val
            reg = np.array(dec2bin(result, cols))
            data[6] += 1
        case 7: # NOT
            reg_val = bin2dec(reg, cols)
            mask = (1 << cols) - 1  # all 1s in the bit width
            result = reg_val ^ mask  # XOR with all 1s = NOT
            reg = np.array(dec2bin(result, cols))
            data[7] += 1



Pvalue = bin2dec_signed(mem[P], cols)
print(f"The value of P is: {Pvalue}")
print(f"Significant ADDs (P = P + A): {sig_add_count}")
print(f"Total significant partial-product ops: {sig_pp_count}")
print(f"\nLOAD: {data[0]}, \nSTORE: {data[1]}, \nADD: {data[2]}, \nBNZ: {data[3]}, \nAND: {data[4]}, \nNAND: {data[5]}, \nXOR: {data[6]}, \nNOT: {data[7]}, \nTotal Instructions: {data[9]}")
