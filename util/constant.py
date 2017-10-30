from model.arguments import get_args


args = get_args()


SYMBOL_GO = '#go#'
GO_ID = args.event_size + 1
SYMBOL_PAD = '#pad#'
PAD_ID = args.event_size + 2
SYMBOL_START = '#bos#'
START_ID = args.event_size + 3
SYMBOL_END = '#eos#'
END_ID = args.event_size + 4

NUM_SPEC_MARK = 5
