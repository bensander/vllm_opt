from typing import List
from typing import Sequence as GenericSequence
from typing import Union

from vllm.sequence import PoolerOutput, SamplerOutput, SequenceGroupOutput


def create_output_by_sequence_group(
        outputs: GenericSequence[Union[SamplerOutput, PoolerOutput]],
        num_seq_groups: int) -> List[List[SequenceGroupOutput]]:
    """Helper method which transforms a 2d list organized by
    [step][sequence group] into [sequence group][step].
    """
    return [list(x) for x in zip(*outputs)]
