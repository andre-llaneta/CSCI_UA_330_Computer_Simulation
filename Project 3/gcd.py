from cpu_core import LOAD, STORE, ADD, BNZ, AND, SHL, SHR, SUB, set_instr


def emit_gcd(mem, start, a, return_line):
    """
    Emits Euclid's algorithm using repeated-subtraction modulo.

    Inputs:
        a['ARG0'], a['ARG1']
    Output:
        a['OUT0']
    Assumes nonnegative inputs.
    Returns:
        next free line after the routine
    """
    pc = start

    # Copy inputs into local working storage
    set_instr(mem, pc, LOAD, a['ARG0']); pc += 1
    set_instr(mem, pc, STORE, a['GCD_A']); pc += 1
    set_instr(mem, pc, LOAD, a['ARG1']); pc += 1
    set_instr(mem, pc, STORE, a['GCD_B']); pc += 1

    loop = pc
    set_instr(mem, pc, LOAD, a['GCD_B']); pc += 1
    set_instr(mem, pc, BNZ, start + 10); pc += 1  # PREP_MOD starts at start+10

    # B == 0 -> answer is A
    set_instr(mem, pc, LOAD, a['GCD_A']); pc += 1
    set_instr(mem, pc, STORE, a['OUT0']); pc += 1
    set_instr(mem, pc, LOAD, a['ONE']); pc += 1
    set_instr(mem, pc, BNZ, return_line); pc += 1

    # PREP_MOD
    set_instr(mem, pc, LOAD, a['GCD_A']); pc += 1
    set_instr(mem, pc, STORE, a['GCD_WORK']); pc += 1

    mod_loop = pc
    set_instr(mem, pc, LOAD, a['GCD_WORK']); pc += 1
    set_instr(mem, pc, SUB, a['GCD_B']); pc += 1
    set_instr(mem, pc, AND, a['SIGN_MASK']); pc += 1
    mod_done_branch = pc
    set_instr(mem, pc, BNZ, 0); pc += 1

    set_instr(mem, pc, LOAD, a['GCD_WORK']); pc += 1
    set_instr(mem, pc, SUB, a['GCD_B']); pc += 1
    set_instr(mem, pc, STORE, a['GCD_WORK']); pc += 1
    set_instr(mem, pc, LOAD, a['ONE']); pc += 1
    set_instr(mem, pc, BNZ, mod_loop); pc += 1

    mod_done = pc
    set_instr(mem, mod_done_branch, BNZ, mod_done)
    set_instr(mem, pc, LOAD, a['GCD_B']); pc += 1
    set_instr(mem, pc, STORE, a['GCD_A']); pc += 1
    set_instr(mem, pc, LOAD, a['GCD_WORK']); pc += 1
    set_instr(mem, pc, STORE, a['GCD_B']); pc += 1
    set_instr(mem, pc, LOAD, a['ONE']); pc += 1
    set_instr(mem, pc, BNZ, loop); pc += 1

    return pc
