//! Spur Test Suite
//!
//! Test numbering mirrors Slurm's testsuite for 1-1 mapping:
//!
//! | Group | Slurm Command | Spur Command | Module |
//! |-------|---------------|--------------|--------|
//! | 1     | srun          | spur run     | t01_run |
//! | 2     | scontrol show | spur show    | t02_show |
//! | 3     | scontrol admin| spur ctl     | t03_admin |
//! | 4     | sinfo         | spur nodes   | t04_nodes |
//! | 5     | squeue        | spur queue   | t05_queue |
//! | 6     | scancel       | spur cancel  | t06_cancel |
//! | 7     | scheduling    | scheduler    | t07_sched |
//! | 12    | sacct         | spur history | t12_history |
//! | 17    | sbatch        | spur submit  | t17_submit |
//! | 21    | sacctmgr      | accounting   | t21_acctmgr |
//! | 24    | priority      | fair-share   | t24_priority |
//! | 28    | job arrays    | arrays       | t28_arrays |
//! | 39    | GPU/GRES      | GPU/GRES     | t39_gpu |
//!
//! Additional Spur-specific test groups:
//! | 50    | core types    | —            | t50_core |
//! | 51    | hostlist      | —            | t51_hostlist |
//! | 52    | config        | —            | t52_config |
//! | 53    | WAL/state     | —            | t53_state |
//! | 54    | REST API      | —            | t54_rest |
//! | 55    | CLI format    | —            | t55_format |
//! | 56    | FFI           | —            | t56_ffi |
//! | 57    | SPANK         | —            | t57_spank |

pub mod harness;

// Unit / component tests (no running daemons needed)
pub mod t05_queue;
pub mod t06_cancel;
pub mod t07_sched;
pub mod t17_submit;
pub mod t21_acctmgr;
pub mod t24_priority;
pub mod t28_arrays;
pub mod t39_gpu;
pub mod t50_core;
pub mod t51_hostlist;
pub mod t52_config;
pub mod t53_state;
pub mod t55_format;
pub mod t56_deps;
pub mod t57_auth;
pub mod t28b_array_expand;
