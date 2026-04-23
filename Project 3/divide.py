from cpu_core import LOAD, STORE, ADD, BNZ, AND, SHL, SHR, SUB, set_instr


def emit_divide(mem, start, a, return_line):
    """
    Emits integer division by repeated subtraction.

    Inputs:
        a['ARG0'] = dividend
        a['ARG1'] = divisor
    Outputs:
        a['OUT0'] = quotient
        a['OUT1'] = remainder
    Assumes positive dividend and positive nonzero divisor.
    Clobbers:
        a['DIV_Q'], a['DIV_R']
    Returns:
        next free line after the routine
    """
    pc = start

    set_instr(mem, pc, LOAD, a['ZERO']); pc += 1
    set_instr(mem, pc, STORE, a['DIV_Q']); pc += 1
    set_instr(mem, pc, LOAD, a['ARG0']); pc += 1
    set_instr(mem, pc, STORE, a['DIV_R']); pc += 1

    loop = pc
    set_instr(mem, pc, LOAD, a['DIV_R']); pc += 1
    set_instr(mem, pc, SUB, a['ARG1']); pc += 1
    set_instr(mem, pc, AND, a['SIGN_MASK']); pc += 1
    done_branch = pc
    set_instr(mem, pc, BNZ, 0); pc += 1  # patched to DONE

    set_instr(mem, pc, LOAD, a['DIV_R']); pc += 1
    set_instr(mem, pc, SUB, a['ARG1']); pc += 1
    set_instr(mem, pc, STORE, a['DIV_R']); pc += 1
    set_instr(mem, pc, LOAD, a['DIV_Q']); pc += 1
    set_instr(mem, pc, ADD, a['ONE']); pc += 1
    set_instr(mem, pc, STORE, a['DIV_Q']); pc += 1
    set_instr(mem, pc, LOAD, a['ONE']); pc += 1
    set_instr(mem, pc, BNZ, loop); pc += 1

    done = pc
    set_instr(mem, done_branch, BNZ, done)
    set_instr(mem, pc, LOAD, a['DIV_Q']); pc += 1
    set_instr(mem, pc, STORE, a['OUT0']); pc += 1
    set_instr(mem, pc, LOAD, a['DIV_R']); pc += 1
    set_instr(mem, pc, STORE, a['OUT1']); pc += 1
    set_instr(mem, pc, LOAD, a['ONE']); pc += 1
    set_instr(mem, pc, BNZ, return_line); pc += 1

    return pc
