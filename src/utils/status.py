from enum import Enum, unique

@unique
class Status(Enum):
    
    SUCCESS = 500

    RANSAC_VOTING_FAIL = 501
    CIR_REFINE_FAIL = 502