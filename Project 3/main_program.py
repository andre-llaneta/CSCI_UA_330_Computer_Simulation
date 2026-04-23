import math

from cpu_core import (
    COLS,
    ROWS,
    LOAD,
    STORE,
    BNZ,
    blank_memory,
    dec2bin,
    dec2bin_signed,
    set_instr,
    run_cpu,
    bin2dec_signed,
    format_counts,
)
from multiply import emit_multiply
from gcd import emit_gcd
from divide import emit_divide

# Shared memory map
A = {
    'ZERO': 0,
    'ONE': 8000,
    'SIGN_MASK': 8001,
    'CONST_16': 8002,
    'INPUT_A': 8003,
    'INPUT_B': 8004,
    'INPUT_C': 8005,
    'INPUT_D': 8006,

    'ARG0': 8010,
    'ARG1': 8011,
    'OUT0': 8012,
    'OUT1': 8013,

    'RESULT_MUL': 8014,
    'RESULT_GCD': 8015,
    'RESULT_DIV_Q': 8016,
    'RESULT_DIV_R': 8017,

    'MUL_A': 8020,
    'MUL_B': 8021,
    'MUL_P': 8022,
    'MUL_CNT': 8023,

    'GCD_A': 8030,
    'GCD_B': 8031,
    'GCD_WORK': 8032,

    'DIV_Q': 8040,
    'DIV_R': 8041,
}

MUL_START = 200
GCD_START = 300
DIV_START = 400

# Edit these four values to run: (a * b) / gcd(c, d)
INPUT_A = 30
INPUT_B = 18
INPUT_C = 630
INPUT_D = 90


def init_data(mem):
    mem[A['ZERO']] = dec2bin(0, COLS)
    mem[A['ONE']] = dec2bin(1, COLS)
    mem[A['SIGN_MASK']] = dec2bin(1 << 15, COLS)
    mem[A['CONST_16']] = dec2bin(16, COLS)
    mem[A['INPUT_A']] = dec2bin_signed(INPUT_A, COLS)
    mem[A['INPUT_B']] = dec2bin_signed(INPUT_B, COLS)
    mem[A['INPUT_C']] = dec2bin_signed(INPUT_C, COLS)
    mem[A['INPUT_D']] = dec2bin_signed(INPUT_D, COLS)

def write_main(mem):
    # Expression: (a * b) / gcd(c, d)
    # multiply first
    set_instr(mem, 1, LOAD, A['INPUT_A'])
    set_instr(mem, 2, STORE, A['ARG0'])
    set_instr(mem, 3, LOAD, A['INPUT_B'])
    set_instr(mem, 4, STORE, A['ARG1'])
    set_instr(mem, 5, LOAD, A['ONE'])
    set_instr(mem, 6, BNZ, MUL_START)

    # continuation after multiply
    set_instr(mem, 7, LOAD, A['OUT0'])
    set_instr(mem, 8, STORE, A['RESULT_MUL'])

    # gcd next
    set_instr(mem, 9, LOAD, A['INPUT_C'])
    set_instr(mem, 10, STORE, A['ARG0'])
    set_instr(mem, 11, LOAD, A['INPUT_D'])
    set_instr(mem, 12, STORE, A['ARG1'])
    set_instr(mem, 13, LOAD, A['ONE'])
    set_instr(mem, 14, BNZ, GCD_START)

    # continuation after gcd
    set_instr(mem, 15, LOAD, A['OUT0'])
    set_instr(mem, 16, STORE, A['RESULT_GCD'])

    # divide RESULT_MUL by RESULT_GCD
    set_instr(mem, 17, LOAD, A['RESULT_MUL'])
    set_instr(mem, 18, STORE, A['ARG0'])
    set_instr(mem, 19, LOAD, A['RESULT_GCD'])
    set_instr(mem, 20, STORE, A['ARG1'])
    set_instr(mem, 21, LOAD, A['ONE'])
    set_instr(mem, 22, BNZ, DIV_START)

    # continuation after divide
    set_instr(mem, 23, LOAD, A['OUT0'])
    set_instr(mem, 24, STORE, A['RESULT_DIV_Q'])
    set_instr(mem, 25, LOAD, A['OUT1'])
    set_instr(mem, 26, STORE, A['RESULT_DIV_R'])

    # halt
    set_instr(mem, 27, LOAD, A['ONE'])
    set_instr(mem, 28, BNZ, A['ZERO'])


def build_program(mem):
    write_main(mem)
    emit_multiply(mem, MUL_START, A, return_line=7)
    emit_gcd(mem, GCD_START, A, return_line=15)
    emit_divide(mem, DIV_START, A, return_line=23)


def main():
    mem = blank_memory()
    init_data(mem)
    build_program(mem)

    mem, data, _ = run_cpu(mem)

    mul_value = bin2dec_signed(mem[A['RESULT_MUL']], COLS)
    gcd_value = bin2dec_signed(mem[A['RESULT_GCD']], COLS)
    q_value = bin2dec_signed(mem[A['RESULT_DIV_Q']], COLS)
    r_value = bin2dec_signed(mem[A['RESULT_DIV_R']], COLS)

    expected_mul = INPUT_A * INPUT_B
    expected_gcd = math.gcd(INPUT_C, INPUT_D)
    expected_q = expected_mul // expected_gcd
    expected_r = expected_mul % expected_gcd

    print("Composed arithmetic program on the simulated computer")
    print(f"Expression: ({INPUT_A} * {INPUT_B}) / gcd({INPUT_C}, {INPUT_D})")
    print()
    print(f"Multiply result: {mul_value} (expected {expected_mul})")
    print(f"GCD result: {gcd_value} (expected {expected_gcd})")
    print(f"Division result: quotient = {q_value}, remainder = {r_value}")
    print(f"Expected: quotient = {expected_q}, remainder = {expected_r}")
    print()
    print(format_counts(data))


if __name__ == "__main__":
    main()
