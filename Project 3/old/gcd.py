# gcd
import numpy as np

def bin2dec(arr, n):
    dec = 0
    for i in range(n):
        dec += int(arr[i]) * (2 ** i)
    return int(dec)

def bin2dec_signed(arr, n):
    if arr[n - 1] == 1:
        inverted = [(1 - int(bit)) for bit in arr]
        magnitude = bin2dec(inverted, n) + 1
        return -magnitude
    else:
        return bin2dec(arr, n)

def dec2bin(dec, n):
    bin_arr = [0] * n
    for i in range(n):
        bin_arr[i] = dec % 2
        dec = dec // 2
    return bin_arr

def dec2bin_signed(dec, n):
    if dec >= 0:
        return dec2bin(dec, n)
    else:
        magnitude = -dec
        bin_mag = dec2bin(magnitude, n)
        inverted = [(1 - bit) for bit in bin_mag]
        carry = 1
        for i in range(n):
            inverted[i] += carry
            if inverted[i] > 1:
                inverted[i] = 0
                carry = 1
            else:
                carry = 0
        return inverted

'''
000 - LOAD  mem[address] -> reg
001 - STORE reg -> mem[address]
010 - ADD   reg + mem[address] -> reg
011 - BNZ   if reg != 0, PC = address
100 - AND   reg & mem[address] -> reg
101 - SHL   reg << mem[address] (shift left logical)
110 - SHR   reg >> mem[address] (shift right arithmetic)
111 - SUB   reg - mem[address] -> reg (2's complement subtraction)
'''

# Memory addresses
ZERO       = 0
A          = 102   # first number for GCD
B          = 103   # second number for GCD
P          = 104   # result (GCD)
ONE        = 106
SHIFT_DIST = 107   # used for computing sign mask
TEMP       = 108   # temporary storage for swaps
SIGN_MASK  = 109   # sign bit mask (0x8000)

cols, rows = 16, 2**13
mem = np.zeros((rows, cols))
reg = np.zeros(cols)
pc = 1

data = np.zeros(10)

# Initialize constants
mem[ZERO]       = dec2bin(0, cols)
mem[ONE]        = dec2bin(1, cols)
mem[SHIFT_DIST] = dec2bin(15, cols)  # shift by 15 to compute sign mask

# Test values
Avalue = 48
Bvalue = 18
mem[A] = dec2bin_signed(Avalue, cols)
mem[B] = dec2bin_signed(Bvalue, cols)
mem[P] = dec2bin(0, cols)


# EUCLIDEAN GCD ALGORITHM
# Initialization (lines 1-3): Compute sign mask = 1 << 15 = 0x8000
mem[1, :] = dec2bin(0, 3) + dec2bin(ONE, 13)           # LOAD 1
mem[2, :] = dec2bin(5, 3) + dec2bin(SHIFT_DIST, 13)   # SHL 15
mem[3, :] = dec2bin(1, 3) + dec2bin(SIGN_MASK, 13)    # STORE sign mask

# Main loop (line 5): Check if B == 0
mem[5, :] = dec2bin(0, 3) + dec2bin(B, 13)          # LOAD B
mem[6, :] = dec2bin(3, 3) + dec2bin(10, 13)         # BNZ to MOD_LOOP (B != 0)
# B is 0, so GCD = A
mem[7, :] = dec2bin(0, 3) + dec2bin(A, 13)          # LOAD A
mem[8, :] = dec2bin(1, 3) + dec2bin(P, 13)          # STORE P (result)
mem[9, :] = dec2bin(0, 3) + dec2bin(ONE, 13)        # LOAD 1
mem[9, :] = dec2bin(3, 3) + dec2bin(0, 13)          # BNZ 0 (halt)

# MOD_LOOP (line 10): Compute A mod B using repeated subtraction
# First check if A - B is negative (using sign bit AND)
mem[10, :] = dec2bin(0, 3) + dec2bin(A, 13)         # LOAD A
mem[11, :] = dec2bin(7, 3) + dec2bin(B, 13)         # SUB B (reg = A - B)
mem[12, :] = dec2bin(4, 3) + dec2bin(SIGN_MASK, 13) # AND sign mask (check if < 0)
mem[13, :] = dec2bin(3, 3) + dec2bin(20, 13)        # BNZ to SWAP (if A < B)
# A >= B, so update A = A - B and loop back
mem[14, :] = dec2bin(0, 3) + dec2bin(A, 13)         # LOAD A
mem[15, :] = dec2bin(7, 3) + dec2bin(B, 13)         # SUB B
mem[16, :] = dec2bin(1, 3) + dec2bin(A, 13)         # STORE A (A = A - B)
mem[17, :] = dec2bin(0, 3) + dec2bin(ONE, 13)       # LOAD 1 (set non-zero for BNZ)
mem[18, :] = dec2bin(3, 3) + dec2bin(10, 13)        # BNZ to MOD_LOOP (loop)

# SWAP (line 20): Swap A and B (A ← B, B ← A mod B)
mem[20, :] = dec2bin(0, 3) + dec2bin(A, 13)         # LOAD A (A mod B)
mem[21, :] = dec2bin(1, 3) + dec2bin(TEMP, 13)      # STORE TEMP (temp = A mod B)
mem[22, :] = dec2bin(0, 3) + dec2bin(B, 13)         # LOAD B
mem[23, :] = dec2bin(1, 3) + dec2bin(A, 13)         # STORE A (A = B)
mem[24, :] = dec2bin(0, 3) + dec2bin(TEMP, 13)      # LOAD TEMP
mem[25, :] = dec2bin(1, 3) + dec2bin(B, 13)         # STORE B (B = A mod B)
mem[26, :] = dec2bin(0, 3) + dec2bin(ONE, 13)       # LOAD 1
mem[27, :] = dec2bin(3, 3) + dec2bin(5, 13)         # BNZ to MAIN_LOOP


while pc > 0:
    instr = mem[pc]
    opcode = bin2dec(instr[0:3], 3)
    address = bin2dec(instr[3:16], 13)
    data[9] += 1

    match opcode:
        case 0: # LOAD
            reg = mem[address].copy()
            data[0] += 1
        case 1: # STORE
            mem[address] = reg.copy()
            data[1] += 1
        case 2: # ADD
            reg_val = bin2dec_signed(reg, cols)
            mem_val = bin2dec_signed(mem[address], cols)
            result = reg_val + mem_val
            if result < -(2 ** (cols - 1)):
                result = result + (2 ** cols)
            elif result >= (2 ** (cols - 1)):
                result = result - (2 ** cols)
            reg = np.array(dec2bin_signed(result, cols))
            data[2] += 1
        case 3: # BNZ
            reg_val = bin2dec_signed(reg, cols)
            if reg_val != 0:
                pc = address
                continue
            data[3] += 1
        case 4: # AND
            reg_val = bin2dec(reg, cols)
            mem_val = bin2dec(mem[address], cols)
            result = reg_val & mem_val
            reg = np.array(dec2bin(result, cols))
            data[4] += 1
        case 5: # SHL
            reg_val = bin2dec_signed(reg, cols)
            shift_amount = bin2dec(mem[address], cols)
            result = (reg_val << shift_amount) & ((1 << cols) - 1)
            reg = np.array(dec2bin_signed(result, cols))
            data[5] += 1
        case 6: # SHR (arithmetic)
            reg_val = bin2dec_signed(reg, cols)
            shift_amount = bin2dec(mem[address], cols)
            if reg_val < 0:
                result = reg_val >> shift_amount
                sign_mask = ((1 << shift_amount) - 1) << (cols - shift_amount) if shift_amount < cols else 0
                result = result | sign_mask
            else:
                result = reg_val >> shift_amount
            result = result & ((1 << cols) - 1)
            reg = np.array(dec2bin_signed(result, cols))
            data[6] += 1
        case 7: # SUB
            reg_val = bin2dec_signed(reg, cols)
            mem_val = bin2dec_signed(mem[address], cols)
            result = reg_val - mem_val
            if result < -(2 ** (cols - 1)):
                result = result + (2 ** cols)
            elif result >= (2 ** (cols - 1)):
                result = result - (2 ** cols)
            reg = np.array(dec2bin_signed(result, cols))
            data[7] += 1

    pc = (pc + 1) % rows


Pvalue = bin2dec_signed(mem[P], cols)
expected = Avalue * Bvalue
print(f"Result: P = {Pvalue}")
print(f"Expected: {expected}")
print(f"\nLOAD: {data[0]}, \nSTORE: {data[1]}, \nADD: {data[2]}, \nBNZ: {data[3]}, \nAND: {data[4]}, \nSHL: {data[5]}, \nSHR: {data[6]}, \nSUB: {data[7]}, \nTotal Instructions: {data[9]}")
