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
ZERO      = 0
A         = 102   # multiplicand
B         = 103   # multiplier
P         = 104   # product (accumulator)
COUNTER   = 105   # iteration counter
ONE       = 106
SHIFT_DIST = 107
TEMP      = 108   # stores current LSB of B (CURR_BIT)
PREV_BIT  = 109   # stores previous LSB of B, init 0

cols, rows = 16, 2**13
mem = np.zeros((rows, cols))
reg = np.zeros(cols)
pc = 1

data = np.zeros(10)

# Initialize constants
mem[ZERO]       = dec2bin(0, cols)
mem[ONE]        = dec2bin(1, cols)
mem[SHIFT_DIST] = dec2bin(1, cols)

# Test values
Avalue = 31
Bvalue = 31
mem[A]       = dec2bin_signed(Avalue, cols)
mem[B]       = dec2bin_signed(Bvalue, cols)
mem[P]       = dec2bin(0, cols)
mem[COUNTER] = dec2bin(16, cols)


# BOOTH'S ALGORITHM
# Setup (lines 1-4)
mem[1, :] = dec2bin(0, 3) + dec2bin(ZERO, 13)      # LOAD 0
mem[2, :] = dec2bin(1, 3) + dec2bin(P, 13)          # STORE P = 0
mem[3, :] = dec2bin(0, 3) + dec2bin(ZERO, 13)       # LOAD 0
mem[4, :] = dec2bin(1, 3) + dec2bin(PREV_BIT, 13)   # STORE PREV_BIT = 0

# MAIN_LOOP (line 5)
mem[5, :] = dec2bin(0, 3) + dec2bin(COUNTER, 13)    # LOAD COUNTER
mem[6, :] = dec2bin(3, 3) + dec2bin(20, 13)         # BNZ to LOOP_BODY (counter != 0)
# Counter == 0: exit
mem[7, :] = dec2bin(0, 3) + dec2bin(ONE, 13)        # LOAD 1
mem[8, :] = dec2bin(3, 3) + dec2bin(0, 13)          # BNZ 0 (halt)

# LOOP_BODY (line 20)
# Get CURR_BIT = LSB of B, compute DIFF = CURR - PREV
mem[20, :] = dec2bin(0, 3) + dec2bin(B, 13)         # LOAD B
mem[21, :] = dec2bin(4, 3) + dec2bin(ONE, 13)       # AND 1 → CURR_BIT
mem[22, :] = dec2bin(1, 3) + dec2bin(TEMP, 13)      # STORE TEMP (save CURR_BIT for PREV update later)
mem[23, :] = dec2bin(7, 3) + dec2bin(PREV_BIT, 13)  # SUB PREV_BIT → reg = DIFF
mem[24, :] = dec2bin(3, 3) + dec2bin(30, 13)        # BNZ to NON_ZERO (if DIFF != 0)
# DIFF == 0: no add/sub needed
mem[25, :] = dec2bin(0, 3) + dec2bin(ONE, 13)       # LOAD 1
mem[26, :] = dec2bin(3, 3) + dec2bin(45, 13)        # BNZ to SHIFT_STEP (unconditional)

# NON_ZERO (line 30): distinguish DIFF=1 (subtract) vs DIFF=-1 (add)
# reg still holds DIFF from line 23 (BNZ doesn't modify reg)
mem[30, :] = dec2bin(2, 3) + dec2bin(ONE, 13)       # ADD 1 → if result==0, DIFF was -1; if !=0, DIFF was 1
mem[31, :] = dec2bin(3, 3) + dec2bin(40, 13)        # BNZ to SUB_STEP (DIFF was 1)
# Fall through: DIFF was -1 → ADD A to P
mem[32, :] = dec2bin(0, 3) + dec2bin(P, 13)         # LOAD P
mem[33, :] = dec2bin(2, 3) + dec2bin(A, 13)         # ADD A
mem[34, :] = dec2bin(1, 3) + dec2bin(P, 13)         # STORE P
mem[35, :] = dec2bin(0, 3) + dec2bin(ONE, 13)       # LOAD 1
mem[36, :] = dec2bin(3, 3) + dec2bin(45, 13)        # BNZ to SHIFT_STEP (unconditional)

# SUB_STEP (line 40): DIFF was 1 → SUBTRACT A from P
mem[40, :] = dec2bin(0, 3) + dec2bin(P, 13)         # LOAD P
mem[41, :] = dec2bin(7, 3) + dec2bin(A, 13)         # SUB A
mem[42, :] = dec2bin(1, 3) + dec2bin(P, 13)         # STORE P
# Fall through to SHIFT_STEP

# SHIFT_STEP (line 45)
# Update PREV_BIT = CURR_BIT (saved in TEMP before the shift)
mem[45, :] = dec2bin(0, 3) + dec2bin(TEMP, 13)      # LOAD TEMP (CURR_BIT)
mem[46, :] = dec2bin(1, 3) + dec2bin(PREV_BIT, 13)  # STORE PREV_BIT

# Left-shift A (multiplicand grows in weight each iteration)
mem[47, :] = dec2bin(0, 3) + dec2bin(A, 13)         # LOAD A
mem[48, :] = dec2bin(5, 3) + dec2bin(SHIFT_DIST, 13) # SHL 1
mem[49, :] = dec2bin(1, 3) + dec2bin(A, 13)         # STORE A

# Right-shift B (arithmetic, so sign extends — handles negative multipliers)
mem[50, :] = dec2bin(0, 3) + dec2bin(B, 13)         # LOAD B
mem[51, :] = dec2bin(6, 3) + dec2bin(SHIFT_DIST, 13) # SHR 1
mem[52, :] = dec2bin(1, 3) + dec2bin(B, 13)         # STORE B

# Decrement counter
mem[53, :] = dec2bin(0, 3) + dec2bin(COUNTER, 13)   # LOAD COUNTER
mem[54, :] = dec2bin(7, 3) + dec2bin(ONE, 13)       # SUB 1
mem[55, :] = dec2bin(1, 3) + dec2bin(COUNTER, 13)   # STORE COUNTER

# Jump back to MAIN_LOOP
mem[56, :] = dec2bin(0, 3) + dec2bin(ONE, 13)       # LOAD 1
mem[57, :] = dec2bin(3, 3) + dec2bin(5, 13)         # BNZ to MAIN_LOOP (line 5)

sig_add_count = 0
sig_sub_count = 0
sig_pp_count = 0


while pc > 0:
    # Count significant Booth partial-product operations
    if pc == 33:
        sig_add_count += 1
        sig_pp_count += 1
    elif pc == 41:
        sig_sub_count += 1
        sig_pp_count += 1

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
print(f"Significant ADDs (P = P + A): {sig_add_count}")
print(f"Significant SUBs (P = P - A): {sig_sub_count}")
print(f"Total significant partial-product ops: {sig_pp_count}")
print(f"\nLOAD: {data[0]}, \nSTORE: {data[1]}, \nADD: {data[2]}, \nBNZ: {data[3]}, \nAND: {data[4]}, \nSHL: {data[5]}, \nSHR: {data[6]}, \nSUB: {data[7]}, \nTotal Instructions: {data[9]}")
