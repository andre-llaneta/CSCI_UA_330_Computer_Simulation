from cpu_core import LOAD, STORE, ADD, BNZ, AND, SHL, SHR, SUB, set_instr


def emit_multiply(mem, start, a, return_line):
    """
    Emits a Booth multiplication routine.

    Inputs:
        a['ARG0'], a['ARG1']
    Output:
        a['OUT0']
    Clobbers:
        a['MUL_A'], a['MUL_B'], a['MUL_P'], a['MUL_CNT']
        a['ARG0'], a['ARG1']   # used internally as PREV_BIT and CURR_BIT scratch
    Returns:
        next free line after the routine
    """
    pc = start

    # Copy inputs into local working storage
    set_instr(mem, pc, LOAD, a['ARG0']); pc += 1
    set_instr(mem, pc, STORE, a['MUL_A']); pc += 1
    set_instr(mem, pc, LOAD, a['ARG1']); pc += 1
    set_instr(mem, pc, STORE, a['MUL_B']); pc += 1

    # P = 0
    set_instr(mem, pc, LOAD, a['ZERO']); pc += 1
    set_instr(mem, pc, STORE, a['MUL_P']); pc += 1

    # CNT = 16
    set_instr(mem, pc, LOAD, a['CONST_16']); pc += 1
    set_instr(mem, pc, STORE, a['MUL_CNT']); pc += 1

    # PREV_BIT = 0  (stored in ARG0 as scratch)
    set_instr(mem, pc, LOAD, a['ZERO']); pc += 1
    set_instr(mem, pc, STORE, a['ARG0']); pc += 1

    # MAIN_LOOP
    loop = pc
    set_instr(mem, pc, LOAD, a['MUL_CNT']); pc += 1
    body_branch = pc
    set_instr(mem, pc, BNZ, 0); pc += 1   # patched to loop body

    # FINISH
    set_instr(mem, pc, LOAD, a['MUL_P']); pc += 1
    set_instr(mem, pc, STORE, a['OUT0']); pc += 1
    set_instr(mem, pc, LOAD, a['ONE']); pc += 1
    set_instr(mem, pc, BNZ, return_line); pc += 1

    # LOOP_BODY
    body = pc
    set_instr(mem, body_branch, BNZ, body)

    # CURR_BIT = LSB(MUL_B), store in ARG1 as scratch
    set_instr(mem, pc, LOAD, a['MUL_B']); pc += 1
    set_instr(mem, pc, AND, a['ONE']); pc += 1
    set_instr(mem, pc, STORE, a['ARG1']); pc += 1   # ARG1 = CURR_BIT

    # DIFF = CURR_BIT - PREV_BIT
    set_instr(mem, pc, SUB, a['ARG0']); pc += 1

    nonzero_branch = pc
    set_instr(mem, pc, BNZ, 0); pc += 1   # patched to NON_ZERO

    # DIFF == 0: skip add/sub, jump to SHIFT_STEP
    set_instr(mem, pc, LOAD, a['ONE']); pc += 1
    shift_jump0 = pc
    set_instr(mem, pc, BNZ, 0); pc += 1   # patched to SHIFT_STEP

    # NON_ZERO:
    nonzero = pc
    set_instr(mem, nonzero_branch, BNZ, nonzero)

    # Distinguish DIFF = -1 vs DIFF = +1
    # After ADD 1:
    #   if DIFF was -1, register becomes 0  -> ADD_STEP
    #   if DIFF was +1, register becomes 2  -> SUB_STEP
    set_instr(mem, pc, ADD, a['ONE']); pc += 1
    sub_branch = pc
    set_instr(mem, pc, BNZ, 0); pc += 1   # patched to SUB_STEP

    # ADD_STEP: P = P + A
    add_step = pc
    set_instr(mem, pc, LOAD, a['MUL_P']); pc += 1
    set_instr(mem, pc, ADD, a['MUL_A']); pc += 1
    set_instr(mem, pc, STORE, a['MUL_P']); pc += 1
    set_instr(mem, pc, LOAD, a['ONE']); pc += 1
    shift_jump1 = pc
    set_instr(mem, pc, BNZ, 0); pc += 1   # patched to SHIFT_STEP

    # SUB_STEP: P = P - A
    sub_step = pc
    set_instr(mem, sub_branch, BNZ, sub_step)
    set_instr(mem, pc, LOAD, a['MUL_P']); pc += 1
    set_instr(mem, pc, SUB, a['MUL_A']); pc += 1
    set_instr(mem, pc, STORE, a['MUL_P']); pc += 1

    # SHIFT_STEP
    shift_step = pc
    set_instr(mem, shift_jump0, BNZ, shift_step)
    set_instr(mem, shift_jump1, BNZ, shift_step)

    # PREV_BIT = CURR_BIT
    set_instr(mem, pc, LOAD, a['ARG1']); pc += 1
    set_instr(mem, pc, STORE, a['ARG0']); pc += 1

    # A <<= 1
    set_instr(mem, pc, LOAD, a['MUL_A']); pc += 1
    set_instr(mem, pc, SHL, a['ONE']); pc += 1
    set_instr(mem, pc, STORE, a['MUL_A']); pc += 1

    # B >>= 1  (arithmetic)
    set_instr(mem, pc, LOAD, a['MUL_B']); pc += 1
    set_instr(mem, pc, SHR, a['ONE']); pc += 1
    set_instr(mem, pc, STORE, a['MUL_B']); pc += 1

    # CNT -= 1
    set_instr(mem, pc, LOAD, a['MUL_CNT']); pc += 1
    set_instr(mem, pc, SUB, a['ONE']); pc += 1
    set_instr(mem, pc, STORE, a['MUL_CNT']); pc += 1

    # Jump back to loop
    set_instr(mem, pc, LOAD, a['ONE']); pc += 1
    set_instr(mem, pc, BNZ, loop); pc += 1

    return pc
