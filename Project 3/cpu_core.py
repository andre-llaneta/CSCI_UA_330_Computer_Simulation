import numpy as np

COLS = 16
ROWS = 2**13

# 3-bit opcodes
LOAD = 0
STORE = 1
ADD = 2
BNZ = 3
AND = 4
SHL = 5
SHR = 6
SUB = 7


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


def wrap16_signed(value, cols=COLS):
    low = -(2 ** (cols - 1))
    high = 2 ** (cols - 1)
    if value < low:
        value += 2 ** cols
    elif value >= high:
        value -= 2 ** cols
    return value


def encode_instr(opcode, address, cols=COLS):
    return np.array(dec2bin(opcode, 3) + dec2bin(address, 13), dtype=int)


def set_instr(mem, line, opcode, address):
    mem[line, :] = encode_instr(opcode, address)


def blank_memory(cols=COLS, rows=ROWS):
    mem = np.zeros((rows, cols), dtype=int)
    return mem


def run_cpu(mem, cols=COLS, rows=ROWS, pc_start=1, max_steps=1_000_000):
    reg = np.zeros(cols, dtype=int)
    pc = pc_start
    data = np.zeros(10, dtype=int)
    steps = 0

    while pc > 0:
        if steps >= max_steps:
            raise RuntimeError(f"CPU exceeded max_steps={max_steps}; possible infinite loop")

        instr = mem[pc]
        opcode = bin2dec(instr[0:3], 3)
        address = bin2dec(instr[3:16], 13)

        pc = (pc + 1) % rows
        data[9] += 1
        steps += 1

        match opcode:
            case 0:  # LOAD
                reg = mem[address].copy()
                data[0] += 1
            case 1:  # STORE
                mem[address] = reg.copy()
                data[1] += 1
            case 2:  # ADD
                reg_val = bin2dec_signed(reg, cols)
                mem_val = bin2dec_signed(mem[address], cols)
                result = wrap16_signed(reg_val + mem_val, cols)
                reg = np.array(dec2bin_signed(result, cols), dtype=int)
                data[2] += 1
            case 3:  # BNZ
                if np.any(reg):
                    pc = address
                data[3] += 1
            case 4:  # AND
                reg_val = bin2dec(reg, cols)
                mem_val = bin2dec(mem[address], cols)
                result = reg_val & mem_val
                reg = np.array(dec2bin(result, cols), dtype=int)
                data[4] += 1
            case 5:  # SHL
                reg_val = bin2dec_signed(reg, cols)
                shift_amount = bin2dec(mem[address], cols)
                result = (reg_val << shift_amount) & ((1 << cols) - 1)
                reg = np.array(dec2bin_signed(result, cols), dtype=int)
                data[5] += 1
            case 6:  # SHR (arithmetic)
                reg_val = bin2dec_signed(reg, cols)
                shift_amount = bin2dec(mem[address], cols)
                result = reg_val >> shift_amount
                result = result & ((1 << cols) - 1)
                reg = np.array(dec2bin_signed(result, cols), dtype=int)
                data[6] += 1
            case 7:  # SUB
                reg_val = bin2dec_signed(reg, cols)
                mem_val = bin2dec_signed(mem[address], cols)
                result = wrap16_signed(reg_val - mem_val, cols)
                reg = np.array(dec2bin_signed(result, cols), dtype=int)
                data[7] += 1
            case _:
                raise ValueError(f"Unknown opcode {opcode} at PC={pc}")

    return mem, data, reg


def format_counts(data):
    labels = ["LOAD", "STORE", "ADD", "BNZ", "AND", "SHL", "SHR", "SUB", "UNUSED", "Total Instructions"]
    lines = []
    for i, label in enumerate(labels):
        if label == "UNUSED":
            continue
        lines.append(f"{label}: {int(data[i])}")
    return "\n".join(lines)
