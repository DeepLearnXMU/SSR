from typing import Literal, Optional
from dataclasses import dataclass, field


@dataclass
class GeneralArguments:
    r"""
    Arguments pertaining to which stage we are going to perform.
    """
    stage: Optional[Literal["pt", "sft", "sftrp", "sftreg", "rm", "ppo", "dpo"]] = field(
        default="sft",
        metadata={"help": "Which stage will be performed in training."}
    )
    reg_cl_method: Optional[Literal["ewc", "l2"]]= field(
        default="ewc",
        metadata={"help": ""}
    )
    reg_p: float = field(
        default=0.1
    )