#include <tree_sitter/parser.h>

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#endif

#ifdef _MSC_VER
#pragma optimize("", off)
#elif defined(__clang__)
#pragma clang optimize off
#elif defined(__GNUC__)
#pragma GCC optimize ("O0")
#endif

#define LANGUAGE_VERSION 14
#define STATE_COUNT 270
#define LARGE_STATE_COUNT 2
#define SYMBOL_COUNT 90
#define ALIAS_COUNT 0
#define TOKEN_COUNT 65
#define EXTERNAL_TOKEN_COUNT 0
#define FIELD_COUNT 0
#define MAX_ALIAS_SEQUENCE_LENGTH 15
#define PRODUCTION_ID_COUNT 1

enum {
  sym_comment = 1,
  sym_ws = 2,
  sym_lparen = 3,
  sym_rparen = 4,
  anon_sym_COMMA = 5,
  anon_sym_set_DASHoption = 6,
  anon_sym_datatype = 7,
  anon_sym_sort = 8,
  anon_sym_function = 9,
  anon_sym_COLONunextractable = 10,
  anon_sym_COLONon_merge = 11,
  anon_sym_COLONmerge = 12,
  anon_sym_COLONdefault = 13,
  anon_sym_declare = 14,
  anon_sym_relation = 15,
  anon_sym_ruleset = 16,
  anon_sym_rule = 17,
  anon_sym_COLONruleset = 18,
  anon_sym_COLONname = 19,
  anon_sym_rewrite = 20,
  anon_sym_COLONwhen = 21,
  anon_sym_birewrite = 22,
  anon_sym_let = 23,
  anon_sym_run = 24,
  anon_sym_COLONuntil = 25,
  anon_sym_simplify = 26,
  anon_sym_calc = 27,
  anon_sym_query_DASHextract = 28,
  anon_sym_COLONvariants = 29,
  anon_sym_check = 30,
  anon_sym_check_DASHproof = 31,
  anon_sym_run_DASHschedule = 32,
  anon_sym_print_DASHstats = 33,
  anon_sym_push = 34,
  anon_sym_pop = 35,
  anon_sym_print_DASHfunction = 36,
  anon_sym_print_DASHsize = 37,
  anon_sym_input = 38,
  anon_sym_output = 39,
  anon_sym_fail = 40,
  anon_sym_include = 41,
  anon_sym_saturate = 42,
  anon_sym_seq = 43,
  anon_sym_repeat = 44,
  anon_sym_COLONcost = 45,
  anon_sym_set = 46,
  anon_sym_delete = 47,
  anon_sym_union = 48,
  anon_sym_panic = 49,
  anon_sym_extract = 50,
  anon_sym_LBRACK = 51,
  anon_sym_RBRACK = 52,
  anon_sym_EQ = 53,
  sym_type = 54,
  anon_sym_true = 55,
  anon_sym_false = 56,
  sym_num = 57,
  sym_unum = 58,
  anon_sym_NaN = 59,
  aux_sym_f64_token1 = 60,
  anon_sym_inf = 61,
  anon_sym_DASHinf = 62,
  sym_ident = 63,
  sym_string = 64,
  sym_source_file = 65,
  sym_command = 66,
  sym_schedule = 67,
  sym_cost = 68,
  sym_nonletaction = 69,
  sym_action = 70,
  sym_fact = 71,
  sym_schema = 72,
  sym_expr = 73,
  sym_literal = 74,
  sym_callexpr = 75,
  sym_variant = 76,
  sym_identsort = 77,
  sym_unit = 78,
  sym_bool = 79,
  sym_f64 = 80,
  sym_symstring = 81,
  aux_sym_source_file_repeat1 = 82,
  aux_sym_command_repeat1 = 83,
  aux_sym_command_repeat2 = 84,
  aux_sym_command_repeat3 = 85,
  aux_sym_command_repeat4 = 86,
  aux_sym_command_repeat5 = 87,
  aux_sym_command_repeat6 = 88,
  aux_sym_command_repeat7 = 89,
};

static const char * const ts_symbol_names[] = {
  [ts_builtin_sym_end] = "end",
  [sym_comment] = "comment",
  [sym_ws] = "ws",
  [sym_lparen] = "lparen",
  [sym_rparen] = "rparen",
  [anon_sym_COMMA] = ",",
  [anon_sym_set_DASHoption] = "set-option",
  [anon_sym_datatype] = "datatype",
  [anon_sym_sort] = "sort",
  [anon_sym_function] = "function",
  [anon_sym_COLONunextractable] = ":unextractable",
  [anon_sym_COLONon_merge] = ":on_merge",
  [anon_sym_COLONmerge] = ":merge",
  [anon_sym_COLONdefault] = ":default",
  [anon_sym_declare] = "declare",
  [anon_sym_relation] = "relation",
  [anon_sym_ruleset] = "ruleset",
  [anon_sym_rule] = "rule",
  [anon_sym_COLONruleset] = ":ruleset",
  [anon_sym_COLONname] = ":name",
  [anon_sym_rewrite] = "rewrite",
  [anon_sym_COLONwhen] = ":when",
  [anon_sym_birewrite] = "birewrite",
  [anon_sym_let] = "let",
  [anon_sym_run] = "run",
  [anon_sym_COLONuntil] = ":until",
  [anon_sym_simplify] = "simplify",
  [anon_sym_calc] = "calc",
  [anon_sym_query_DASHextract] = "query-extract",
  [anon_sym_COLONvariants] = ":variants",
  [anon_sym_check] = "check",
  [anon_sym_check_DASHproof] = "check-proof",
  [anon_sym_run_DASHschedule] = "run-schedule",
  [anon_sym_print_DASHstats] = "print-stats",
  [anon_sym_push] = "push",
  [anon_sym_pop] = "pop",
  [anon_sym_print_DASHfunction] = "print-function",
  [anon_sym_print_DASHsize] = "print-size",
  [anon_sym_input] = "input",
  [anon_sym_output] = "output",
  [anon_sym_fail] = "fail",
  [anon_sym_include] = "include",
  [anon_sym_saturate] = "saturate",
  [anon_sym_seq] = "seq",
  [anon_sym_repeat] = "repeat",
  [anon_sym_COLONcost] = ":cost",
  [anon_sym_set] = "set",
  [anon_sym_delete] = "delete",
  [anon_sym_union] = "union",
  [anon_sym_panic] = "panic",
  [anon_sym_extract] = "extract",
  [anon_sym_LBRACK] = "[",
  [anon_sym_RBRACK] = "]",
  [anon_sym_EQ] = "=",
  [sym_type] = "type",
  [anon_sym_true] = "true",
  [anon_sym_false] = "false",
  [sym_num] = "num",
  [sym_unum] = "unum",
  [anon_sym_NaN] = "NaN",
  [aux_sym_f64_token1] = "f64_token1",
  [anon_sym_inf] = "inf",
  [anon_sym_DASHinf] = "-inf",
  [sym_ident] = "ident",
  [sym_string] = "string",
  [sym_source_file] = "source_file",
  [sym_command] = "command",
  [sym_schedule] = "schedule",
  [sym_cost] = "cost",
  [sym_nonletaction] = "nonletaction",
  [sym_action] = "action",
  [sym_fact] = "fact",
  [sym_schema] = "schema",
  [sym_expr] = "expr",
  [sym_literal] = "literal",
  [sym_callexpr] = "callexpr",
  [sym_variant] = "variant",
  [sym_identsort] = "identsort",
  [sym_unit] = "unit",
  [sym_bool] = "bool",
  [sym_f64] = "f64",
  [sym_symstring] = "symstring",
  [aux_sym_source_file_repeat1] = "source_file_repeat1",
  [aux_sym_command_repeat1] = "command_repeat1",
  [aux_sym_command_repeat2] = "command_repeat2",
  [aux_sym_command_repeat3] = "command_repeat3",
  [aux_sym_command_repeat4] = "command_repeat4",
  [aux_sym_command_repeat5] = "command_repeat5",
  [aux_sym_command_repeat6] = "command_repeat6",
  [aux_sym_command_repeat7] = "command_repeat7",
};

static const TSSymbol ts_symbol_map[] = {
  [ts_builtin_sym_end] = ts_builtin_sym_end,
  [sym_comment] = sym_comment,
  [sym_ws] = sym_ws,
  [sym_lparen] = sym_lparen,
  [sym_rparen] = sym_rparen,
  [anon_sym_COMMA] = anon_sym_COMMA,
  [anon_sym_set_DASHoption] = anon_sym_set_DASHoption,
  [anon_sym_datatype] = anon_sym_datatype,
  [anon_sym_sort] = anon_sym_sort,
  [anon_sym_function] = anon_sym_function,
  [anon_sym_COLONunextractable] = anon_sym_COLONunextractable,
  [anon_sym_COLONon_merge] = anon_sym_COLONon_merge,
  [anon_sym_COLONmerge] = anon_sym_COLONmerge,
  [anon_sym_COLONdefault] = anon_sym_COLONdefault,
  [anon_sym_declare] = anon_sym_declare,
  [anon_sym_relation] = anon_sym_relation,
  [anon_sym_ruleset] = anon_sym_ruleset,
  [anon_sym_rule] = anon_sym_rule,
  [anon_sym_COLONruleset] = anon_sym_COLONruleset,
  [anon_sym_COLONname] = anon_sym_COLONname,
  [anon_sym_rewrite] = anon_sym_rewrite,
  [anon_sym_COLONwhen] = anon_sym_COLONwhen,
  [anon_sym_birewrite] = anon_sym_birewrite,
  [anon_sym_let] = anon_sym_let,
  [anon_sym_run] = anon_sym_run,
  [anon_sym_COLONuntil] = anon_sym_COLONuntil,
  [anon_sym_simplify] = anon_sym_simplify,
  [anon_sym_calc] = anon_sym_calc,
  [anon_sym_query_DASHextract] = anon_sym_query_DASHextract,
  [anon_sym_COLONvariants] = anon_sym_COLONvariants,
  [anon_sym_check] = anon_sym_check,
  [anon_sym_check_DASHproof] = anon_sym_check_DASHproof,
  [anon_sym_run_DASHschedule] = anon_sym_run_DASHschedule,
  [anon_sym_print_DASHstats] = anon_sym_print_DASHstats,
  [anon_sym_push] = anon_sym_push,
  [anon_sym_pop] = anon_sym_pop,
  [anon_sym_print_DASHfunction] = anon_sym_print_DASHfunction,
  [anon_sym_print_DASHsize] = anon_sym_print_DASHsize,
  [anon_sym_input] = anon_sym_input,
  [anon_sym_output] = anon_sym_output,
  [anon_sym_fail] = anon_sym_fail,
  [anon_sym_include] = anon_sym_include,
  [anon_sym_saturate] = anon_sym_saturate,
  [anon_sym_seq] = anon_sym_seq,
  [anon_sym_repeat] = anon_sym_repeat,
  [anon_sym_COLONcost] = anon_sym_COLONcost,
  [anon_sym_set] = anon_sym_set,
  [anon_sym_delete] = anon_sym_delete,
  [anon_sym_union] = anon_sym_union,
  [anon_sym_panic] = anon_sym_panic,
  [anon_sym_extract] = anon_sym_extract,
  [anon_sym_LBRACK] = anon_sym_LBRACK,
  [anon_sym_RBRACK] = anon_sym_RBRACK,
  [anon_sym_EQ] = anon_sym_EQ,
  [sym_type] = sym_type,
  [anon_sym_true] = anon_sym_true,
  [anon_sym_false] = anon_sym_false,
  [sym_num] = sym_num,
  [sym_unum] = sym_unum,
  [anon_sym_NaN] = anon_sym_NaN,
  [aux_sym_f64_token1] = aux_sym_f64_token1,
  [anon_sym_inf] = anon_sym_inf,
  [anon_sym_DASHinf] = anon_sym_DASHinf,
  [sym_ident] = sym_ident,
  [sym_string] = sym_string,
  [sym_source_file] = sym_source_file,
  [sym_command] = sym_command,
  [sym_schedule] = sym_schedule,
  [sym_cost] = sym_cost,
  [sym_nonletaction] = sym_nonletaction,
  [sym_action] = sym_action,
  [sym_fact] = sym_fact,
  [sym_schema] = sym_schema,
  [sym_expr] = sym_expr,
  [sym_literal] = sym_literal,
  [sym_callexpr] = sym_callexpr,
  [sym_variant] = sym_variant,
  [sym_identsort] = sym_identsort,
  [sym_unit] = sym_unit,
  [sym_bool] = sym_bool,
  [sym_f64] = sym_f64,
  [sym_symstring] = sym_symstring,
  [aux_sym_source_file_repeat1] = aux_sym_source_file_repeat1,
  [aux_sym_command_repeat1] = aux_sym_command_repeat1,
  [aux_sym_command_repeat2] = aux_sym_command_repeat2,
  [aux_sym_command_repeat3] = aux_sym_command_repeat3,
  [aux_sym_command_repeat4] = aux_sym_command_repeat4,
  [aux_sym_command_repeat5] = aux_sym_command_repeat5,
  [aux_sym_command_repeat6] = aux_sym_command_repeat6,
  [aux_sym_command_repeat7] = aux_sym_command_repeat7,
};

static const TSSymbolMetadata ts_symbol_metadata[] = {
  [ts_builtin_sym_end] = {
    .visible = false,
    .named = true,
  },
  [sym_comment] = {
    .visible = true,
    .named = true,
  },
  [sym_ws] = {
    .visible = true,
    .named = true,
  },
  [sym_lparen] = {
    .visible = true,
    .named = true,
  },
  [sym_rparen] = {
    .visible = true,
    .named = true,
  },
  [anon_sym_COMMA] = {
    .visible = true,
    .named = false,
  },
  [anon_sym_set_DASHoption] = {
    .visible = true,
    .named = false,
  },
  [anon_sym_datatype] = {
    .visible = true,
    .named = false,
  },
  [anon_sym_sort] = {
    .visible = true,
    .named = false,
  },
  [anon_sym_function] = {
    .visible = true,
    .named = false,
  },
  [anon_sym_COLONunextractable] = {
    .visible = true,
    .named = false,
  },
  [anon_sym_COLONon_merge] = {
    .visible = true,
    .named = false,
  },
  [anon_sym_COLONmerge] = {
    .visible = true,
    .named = false,
  },
  [anon_sym_COLONdefault] = {
    .visible = true,
    .named = false,
  },
  [anon_sym_declare] = {
    .visible = true,
    .named = false,
  },
  [anon_sym_relation] = {
    .visible = true,
    .named = false,
  },
  [anon_sym_ruleset] = {
    .visible = true,
    .named = false,
  },
  [anon_sym_rule] = {
    .visible = true,
    .named = false,
  },
  [anon_sym_COLONruleset] = {
    .visible = true,
    .named = false,
  },
  [anon_sym_COLONname] = {
    .visible = true,
    .named = false,
  },
  [anon_sym_rewrite] = {
    .visible = true,
    .named = false,
  },
  [anon_sym_COLONwhen] = {
    .visible = true,
    .named = false,
  },
  [anon_sym_birewrite] = {
    .visible = true,
    .named = false,
  },
  [anon_sym_let] = {
    .visible = true,
    .named = false,
  },
  [anon_sym_run] = {
    .visible = true,
    .named = false,
  },
  [anon_sym_COLONuntil] = {
    .visible = true,
    .named = false,
  },
  [anon_sym_simplify] = {
    .visible = true,
    .named = false,
  },
  [anon_sym_calc] = {
    .visible = true,
    .named = false,
  },
  [anon_sym_query_DASHextract] = {
    .visible = true,
    .named = false,
  },
  [anon_sym_COLONvariants] = {
    .visible = true,
    .named = false,
  },
  [anon_sym_check] = {
    .visible = true,
    .named = false,
  },
  [anon_sym_check_DASHproof] = {
    .visible = true,
    .named = false,
  },
  [anon_sym_run_DASHschedule] = {
    .visible = true,
    .named = false,
  },
  [anon_sym_print_DASHstats] = {
    .visible = true,
    .named = false,
  },
  [anon_sym_push] = {
    .visible = true,
    .named = false,
  },
  [anon_sym_pop] = {
    .visible = true,
    .named = false,
  },
  [anon_sym_print_DASHfunction] = {
    .visible = true,
    .named = false,
  },
  [anon_sym_print_DASHsize] = {
    .visible = true,
    .named = false,
  },
  [anon_sym_input] = {
    .visible = true,
    .named = false,
  },
  [anon_sym_output] = {
    .visible = true,
    .named = false,
  },
  [anon_sym_fail] = {
    .visible = true,
    .named = false,
  },
  [anon_sym_include] = {
    .visible = true,
    .named = false,
  },
  [anon_sym_saturate] = {
    .visible = true,
    .named = false,
  },
  [anon_sym_seq] = {
    .visible = true,
    .named = false,
  },
  [anon_sym_repeat] = {
    .visible = true,
    .named = false,
  },
  [anon_sym_COLONcost] = {
    .visible = true,
    .named = false,
  },
  [anon_sym_set] = {
    .visible = true,
    .named = false,
  },
  [anon_sym_delete] = {
    .visible = true,
    .named = false,
  },
  [anon_sym_union] = {
    .visible = true,
    .named = false,
  },
  [anon_sym_panic] = {
    .visible = true,
    .named = false,
  },
  [anon_sym_extract] = {
    .visible = true,
    .named = false,
  },
  [anon_sym_LBRACK] = {
    .visible = true,
    .named = false,
  },
  [anon_sym_RBRACK] = {
    .visible = true,
    .named = false,
  },
  [anon_sym_EQ] = {
    .visible = true,
    .named = false,
  },
  [sym_type] = {
    .visible = true,
    .named = true,
  },
  [anon_sym_true] = {
    .visible = true,
    .named = false,
  },
  [anon_sym_false] = {
    .visible = true,
    .named = false,
  },
  [sym_num] = {
    .visible = true,
    .named = true,
  },
  [sym_unum] = {
    .visible = true,
    .named = true,
  },
  [anon_sym_NaN] = {
    .visible = true,
    .named = false,
  },
  [aux_sym_f64_token1] = {
    .visible = false,
    .named = false,
  },
  [anon_sym_inf] = {
    .visible = true,
    .named = false,
  },
  [anon_sym_DASHinf] = {
    .visible = true,
    .named = false,
  },
  [sym_ident] = {
    .visible = true,
    .named = true,
  },
  [sym_string] = {
    .visible = true,
    .named = true,
  },
  [sym_source_file] = {
    .visible = true,
    .named = true,
  },
  [sym_command] = {
    .visible = true,
    .named = true,
  },
  [sym_schedule] = {
    .visible = true,
    .named = true,
  },
  [sym_cost] = {
    .visible = true,
    .named = true,
  },
  [sym_nonletaction] = {
    .visible = true,
    .named = true,
  },
  [sym_action] = {
    .visible = true,
    .named = true,
  },
  [sym_fact] = {
    .visible = true,
    .named = true,
  },
  [sym_schema] = {
    .visible = true,
    .named = true,
  },
  [sym_expr] = {
    .visible = true,
    .named = true,
  },
  [sym_literal] = {
    .visible = true,
    .named = true,
  },
  [sym_callexpr] = {
    .visible = true,
    .named = true,
  },
  [sym_variant] = {
    .visible = true,
    .named = true,
  },
  [sym_identsort] = {
    .visible = true,
    .named = true,
  },
  [sym_unit] = {
    .visible = true,
    .named = true,
  },
  [sym_bool] = {
    .visible = true,
    .named = true,
  },
  [sym_f64] = {
    .visible = true,
    .named = true,
  },
  [sym_symstring] = {
    .visible = true,
    .named = true,
  },
  [aux_sym_source_file_repeat1] = {
    .visible = false,
    .named = false,
  },
  [aux_sym_command_repeat1] = {
    .visible = false,
    .named = false,
  },
  [aux_sym_command_repeat2] = {
    .visible = false,
    .named = false,
  },
  [aux_sym_command_repeat3] = {
    .visible = false,
    .named = false,
  },
  [aux_sym_command_repeat4] = {
    .visible = false,
    .named = false,
  },
  [aux_sym_command_repeat5] = {
    .visible = false,
    .named = false,
  },
  [aux_sym_command_repeat6] = {
    .visible = false,
    .named = false,
  },
  [aux_sym_command_repeat7] = {
    .visible = false,
    .named = false,
  },
};

static const TSSymbol ts_alias_sequences[PRODUCTION_ID_COUNT][MAX_ALIAS_SEQUENCE_LENGTH] = {
  [0] = {0},
};

static const uint16_t ts_non_terminal_alias_map[] = {
  0,
};

static const TSStateId ts_primary_state_ids[STATE_COUNT] = {
  [0] = 0,
  [1] = 1,
  [2] = 2,
  [3] = 3,
  [4] = 4,
  [5] = 5,
  [6] = 6,
  [7] = 7,
  [8] = 8,
  [9] = 9,
  [10] = 10,
  [11] = 11,
  [12] = 12,
  [13] = 13,
  [14] = 14,
  [15] = 15,
  [16] = 16,
  [17] = 17,
  [18] = 18,
  [19] = 19,
  [20] = 20,
  [21] = 21,
  [22] = 22,
  [23] = 23,
  [24] = 24,
  [25] = 25,
  [26] = 26,
  [27] = 27,
  [28] = 28,
  [29] = 29,
  [30] = 30,
  [31] = 31,
  [32] = 32,
  [33] = 33,
  [34] = 34,
  [35] = 35,
  [36] = 36,
  [37] = 37,
  [38] = 38,
  [39] = 39,
  [40] = 40,
  [41] = 41,
  [42] = 42,
  [43] = 43,
  [44] = 44,
  [45] = 45,
  [46] = 46,
  [47] = 47,
  [48] = 48,
  [49] = 49,
  [50] = 50,
  [51] = 51,
  [52] = 52,
  [53] = 53,
  [54] = 54,
  [55] = 55,
  [56] = 56,
  [57] = 57,
  [58] = 58,
  [59] = 59,
  [60] = 60,
  [61] = 61,
  [62] = 62,
  [63] = 63,
  [64] = 64,
  [65] = 65,
  [66] = 66,
  [67] = 67,
  [68] = 68,
  [69] = 69,
  [70] = 70,
  [71] = 71,
  [72] = 72,
  [73] = 73,
  [74] = 74,
  [75] = 75,
  [76] = 76,
  [77] = 77,
  [78] = 78,
  [79] = 79,
  [80] = 80,
  [81] = 81,
  [82] = 82,
  [83] = 83,
  [84] = 84,
  [85] = 85,
  [86] = 86,
  [87] = 87,
  [88] = 88,
  [89] = 89,
  [90] = 90,
  [91] = 91,
  [92] = 92,
  [93] = 93,
  [94] = 90,
  [95] = 95,
  [96] = 96,
  [97] = 97,
  [98] = 89,
  [99] = 99,
  [100] = 100,
  [101] = 101,
  [102] = 91,
  [103] = 103,
  [104] = 101,
  [105] = 103,
  [106] = 106,
  [107] = 107,
  [108] = 84,
  [109] = 109,
  [110] = 110,
  [111] = 82,
  [112] = 80,
  [113] = 113,
  [114] = 114,
  [115] = 115,
  [116] = 116,
  [117] = 117,
  [118] = 118,
  [119] = 119,
  [120] = 120,
  [121] = 115,
  [122] = 122,
  [123] = 123,
  [124] = 124,
  [125] = 125,
  [126] = 126,
  [127] = 61,
  [128] = 128,
  [129] = 129,
  [130] = 130,
  [131] = 131,
  [132] = 132,
  [133] = 133,
  [134] = 134,
  [135] = 135,
  [136] = 136,
  [137] = 137,
  [138] = 60,
  [139] = 139,
  [140] = 140,
  [141] = 141,
  [142] = 142,
  [143] = 59,
  [144] = 144,
  [145] = 145,
  [146] = 146,
  [147] = 147,
  [148] = 57,
  [149] = 149,
  [150] = 58,
  [151] = 151,
  [152] = 152,
  [153] = 153,
  [154] = 154,
  [155] = 155,
  [156] = 156,
  [157] = 156,
  [158] = 158,
  [159] = 159,
  [160] = 160,
  [161] = 161,
  [162] = 162,
  [163] = 163,
  [164] = 164,
  [165] = 165,
  [166] = 166,
  [167] = 167,
  [168] = 168,
  [169] = 169,
  [170] = 170,
  [171] = 171,
  [172] = 172,
  [173] = 173,
  [174] = 172,
  [175] = 175,
  [176] = 176,
  [177] = 177,
  [178] = 178,
  [179] = 179,
  [180] = 180,
  [181] = 181,
  [182] = 182,
  [183] = 183,
  [184] = 184,
  [185] = 185,
  [186] = 186,
  [187] = 187,
  [188] = 188,
  [189] = 189,
  [190] = 190,
  [191] = 191,
  [192] = 192,
  [193] = 193,
  [194] = 194,
  [195] = 195,
  [196] = 196,
  [197] = 197,
  [198] = 198,
  [199] = 199,
  [200] = 200,
  [201] = 201,
  [202] = 202,
  [203] = 203,
  [204] = 204,
  [205] = 205,
  [206] = 206,
  [207] = 207,
  [208] = 208,
  [209] = 209,
  [210] = 210,
  [211] = 211,
  [212] = 212,
  [213] = 213,
  [214] = 214,
  [215] = 215,
  [216] = 216,
  [217] = 217,
  [218] = 218,
  [219] = 219,
  [220] = 220,
  [221] = 221,
  [222] = 222,
  [223] = 223,
  [224] = 224,
  [225] = 225,
  [226] = 226,
  [227] = 227,
  [228] = 228,
  [229] = 229,
  [230] = 230,
  [231] = 231,
  [232] = 232,
  [233] = 233,
  [234] = 234,
  [235] = 235,
  [236] = 236,
  [237] = 237,
  [238] = 238,
  [239] = 239,
  [240] = 240,
  [241] = 241,
  [242] = 242,
  [243] = 243,
  [244] = 244,
  [245] = 245,
  [246] = 246,
  [247] = 247,
  [248] = 248,
  [249] = 249,
  [250] = 250,
  [251] = 251,
  [252] = 252,
  [253] = 253,
  [254] = 254,
  [255] = 255,
  [256] = 256,
  [257] = 257,
  [258] = 258,
  [259] = 259,
  [260] = 260,
  [261] = 261,
  [262] = 262,
  [263] = 263,
  [264] = 264,
  [265] = 265,
  [266] = 266,
  [267] = 267,
  [268] = 268,
  [269] = 264,
};

static bool ts_lex(TSLexer *lexer, TSStateId state) {
  START_LEXER();
  eof = lexer->eof(lexer);
  switch (state) {
    case 0:
      if (eof) ADVANCE(235);
      if (lookahead == '"') ADVANCE(1);
      if (lookahead == '(') ADVANCE(238);
      if (lookahead == ')') ADVANCE(239);
      if (lookahead == ',') ADVANCE(240);
      if (lookahead == '-') ADVANCE(96);
      if (lookahead == ':') ADVANCE(33);
      if (lookahead == ';') ADVANCE(236);
      if (lookahead == '=') ADVANCE(319);
      if (lookahead == 'N') ADVANCE(10);
      if (lookahead == '[') ADVANCE(317);
      if (lookahead == ']') ADVANCE(318);
      if (lookahead == 'b') ADVANCE(94);
      if (lookahead == 'c') ADVANCE(11);
      if (lookahead == 'd') ADVANCE(12);
      if (lookahead == 'e') ADVANCE(221);
      if (lookahead == 'f') ADVANCE(13);
      if (lookahead == 'i') ADVANCE(124);
      if (lookahead == 'l') ADVANCE(70);
      if (lookahead == 'o') ADVANCE(210);
      if (lookahead == 'p') ADVANCE(24);
      if (lookahead == 'q') ADVANCE(217);
      if (lookahead == 'r') ADVANCE(49);
      if (lookahead == 's') ADVANCE(17);
      if (lookahead == 't') ADVANCE(167);
      if (lookahead == 'u') ADVANCE(134);
      if (lookahead == '\t' ||
          lookahead == '\n' ||
          lookahead == '\r' ||
          lookahead == ' ') ADVANCE(237);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(327);
      END_STATE();
    case 1:
      if (lookahead == '"') ADVANCE(501);
      if (lookahead != 0 &&
          lookahead != '\n' &&
          lookahead != '\\') ADVANCE(1);
      END_STATE();
    case 2:
      if (lookahead == '+') ADVANCE(4);
      if (lookahead == '-') ADVANCE(232);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(332);
      END_STATE();
    case 3:
      if (lookahead == '-') ADVANCE(87);
      END_STATE();
    case 4:
      if (lookahead == '-') ADVANCE(232);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(332);
      END_STATE();
    case 5:
      if (lookahead == '-') ADVANCE(82);
      END_STATE();
    case 6:
      if (lookahead == ';') ADVANCE(236);
      if (lookahead == '=') ADVANCE(320);
      if (lookahead == '\t' ||
          lookahead == '\n' ||
          lookahead == '\r' ||
          lookahead == ' ') ADVANCE(237);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '-' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '_' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('a' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 7:
      if (lookahead == ';') ADVANCE(236);
      if (lookahead == 'r') ADVANCE(79);
      if (lookahead == 's') ADVANCE(18);
      if (lookahead == '\t' ||
          lookahead == '\n' ||
          lookahead == '\r' ||
          lookahead == ' ') ADVANCE(237);
      END_STATE();
    case 8:
      if (lookahead == 'N') ADVANCE(329);
      END_STATE();
    case 9:
      if (lookahead == '_') ADVANCE(123);
      END_STATE();
    case 10:
      if (lookahead == 'a') ADVANCE(8);
      END_STATE();
    case 11:
      if (lookahead == 'a') ADVANCE(109);
      if (lookahead == 'h') ADVANCE(68);
      END_STATE();
    case 12:
      if (lookahead == 'a') ADVANCE(192);
      if (lookahead == 'e') ADVANCE(38);
      END_STATE();
    case 13:
      if (lookahead == 'a') ADVANCE(98);
      if (lookahead == 'u') ADVANCE(135);
      END_STATE();
    case 14:
      if (lookahead == 'a') ADVANCE(31);
      END_STATE();
    case 15:
      if (lookahead == 'a') ADVANCE(122);
      END_STATE();
    case 16:
      if (lookahead == 'a') ADVANCE(159);
      END_STATE();
    case 17:
      if (lookahead == 'a') ADVANCE(193);
      if (lookahead == 'e') ADVANCE(155);
      if (lookahead == 'i') ADVANCE(121);
      if (lookahead == 'o') ADVANCE(161);
      END_STATE();
    case 18:
      if (lookahead == 'a') ADVANCE(193);
      if (lookahead == 'e') ADVANCE(154);
      END_STATE();
    case 19:
      if (lookahead == 'a') ADVANCE(212);
      END_STATE();
    case 20:
      if (lookahead == 'a') ADVANCE(44);
      END_STATE();
    case 21:
      if (lookahead == 'a') ADVANCE(191);
      END_STATE();
    case 22:
      if (lookahead == 'a') ADVANCE(184);
      END_STATE();
    case 23:
      if (lookahead == 'a') ADVANCE(197);
      END_STATE();
    case 24:
      if (lookahead == 'a') ADVANCE(136);
      if (lookahead == 'o') ADVANCE(148);
      if (lookahead == 'r') ADVANCE(100);
      if (lookahead == 'u') ADVANCE(173);
      END_STATE();
    case 25:
      if (lookahead == 'a') ADVANCE(165);
      END_STATE();
    case 26:
      if (lookahead == 'a') ADVANCE(41);
      END_STATE();
    case 27:
      if (lookahead == 'a') ADVANCE(138);
      END_STATE();
    case 28:
      if (lookahead == 'a') ADVANCE(42);
      END_STATE();
    case 29:
      if (lookahead == 'a') ADVANCE(201);
      END_STATE();
    case 30:
      if (lookahead == 'a') ADVANCE(205);
      END_STATE();
    case 31:
      if (lookahead == 'b') ADVANCE(119);
      END_STATE();
    case 32:
      if (lookahead == 'c') ADVANCE(141);
      END_STATE();
    case 33:
      if (lookahead == 'c') ADVANCE(141);
      if (lookahead == 'd') ADVANCE(67);
      if (lookahead == 'm') ADVANCE(71);
      if (lookahead == 'n') ADVANCE(15);
      if (lookahead == 'o') ADVANCE(126);
      if (lookahead == 'r') ADVANCE(209);
      if (lookahead == 'u') ADVANCE(127);
      if (lookahead == 'v') ADVANCE(16);
      if (lookahead == 'w') ADVANCE(91);
      END_STATE();
    case 34:
      if (lookahead == 'c') ADVANCE(141);
      if (lookahead == 'd') ADVANCE(67);
      if (lookahead == 'm') ADVANCE(71);
      if (lookahead == 'n') ADVANCE(15);
      if (lookahead == 'o') ADVANCE(126);
      if (lookahead == 'r') ADVANCE(209);
      if (lookahead == 'u') ADVANCE(127);
      if (lookahead == 'w') ADVANCE(91);
      END_STATE();
    case 35:
      if (lookahead == 'c') ADVANCE(275);
      END_STATE();
    case 36:
      if (lookahead == 'c') ADVANCE(108);
      END_STATE();
    case 37:
      if (lookahead == 'c') ADVANCE(313);
      END_STATE();
    case 38:
      if (lookahead == 'c') ADVANCE(113);
      if (lookahead == 'l') ADVANCE(75);
      END_STATE();
    case 39:
      if (lookahead == 'c') ADVANCE(112);
      if (lookahead == 'f') ADVANCE(333);
      if (lookahead == 'p') ADVANCE(214);
      END_STATE();
    case 40:
      if (lookahead == 'c') ADVANCE(203);
      END_STATE();
    case 41:
      if (lookahead == 'c') ADVANCE(185);
      END_STATE();
    case 42:
      if (lookahead == 'c') ADVANCE(189);
      END_STATE();
    case 43:
      if (lookahead == 'c') ADVANCE(92);
      END_STATE();
    case 44:
      if (lookahead == 'c') ADVANCE(198);
      END_STATE();
    case 45:
      if (lookahead == 'c') ADVANCE(206);
      END_STATE();
    case 46:
      if (lookahead == 'd') ADVANCE(67);
      if (lookahead == 'r') ADVANCE(209);
      if (lookahead == 'v') ADVANCE(16);
      if (lookahead == 'w') ADVANCE(91);
      END_STATE();
    case 47:
      if (lookahead == 'd') ADVANCE(58);
      END_STATE();
    case 48:
      if (lookahead == 'd') ADVANCE(216);
      END_STATE();
    case 49:
      if (lookahead == 'e') ADVANCE(120);
      if (lookahead == 'u') ADVANCE(115);
      END_STATE();
    case 50:
      if (lookahead == 'e') ADVANCE(220);
      END_STATE();
    case 51:
      if (lookahead == 'e') ADVANCE(258);
      END_STATE();
    case 52:
      if (lookahead == 'e') ADVANCE(323);
      END_STATE();
    case 53:
      if (lookahead == 'e') ADVANCE(261);
      END_STATE();
    case 54:
      if (lookahead == 'e') ADVANCE(325);
      END_STATE();
    case 55:
      if (lookahead == 'e') ADVANCE(250);
      END_STATE();
    case 56:
      if (lookahead == 'e') ADVANCE(309);
      END_STATE();
    case 57:
      if (lookahead == 'e') ADVANCE(252);
      END_STATE();
    case 58:
      if (lookahead == 'e') ADVANCE(302);
      END_STATE();
    case 59:
      if (lookahead == 'e') ADVANCE(262);
      END_STATE();
    case 60:
      if (lookahead == 'e') ADVANCE(242);
      END_STATE();
    case 61:
      if (lookahead == 'e') ADVANCE(304);
      END_STATE();
    case 62:
      if (lookahead == 'e') ADVANCE(249);
      END_STATE();
    case 63:
      if (lookahead == 'e') ADVANCE(265);
      END_STATE();
    case 64:
      if (lookahead == 'e') ADVANCE(294);
      END_STATE();
    case 65:
      if (lookahead == 'e') ADVANCE(284);
      END_STATE();
    case 66:
      if (lookahead == 'e') ADVANCE(248);
      END_STATE();
    case 67:
      if (lookahead == 'e') ADVANCE(86);
      END_STATE();
    case 68:
      if (lookahead == 'e') ADVANCE(36);
      END_STATE();
    case 69:
      if (lookahead == 'e') ADVANCE(48);
      END_STATE();
    case 70:
      if (lookahead == 'e') ADVANCE(178);
      END_STATE();
    case 71:
      if (lookahead == 'e') ADVANCE(156);
      END_STATE();
    case 72:
      if (lookahead == 'e') ADVANCE(177);
      END_STATE();
    case 73:
      if (lookahead == 'e') ADVANCE(157);
      END_STATE();
    case 74:
      if (lookahead == 'e') ADVANCE(128);
      END_STATE();
    case 75:
      if (lookahead == 'e') ADVANCE(199);
      END_STATE();
    case 76:
      if (lookahead == 'e') ADVANCE(186);
      END_STATE();
    case 77:
      if (lookahead == 'e') ADVANCE(188);
      END_STATE();
    case 78:
      if (lookahead == 'e') ADVANCE(166);
      END_STATE();
    case 79:
      if (lookahead == 'e') ADVANCE(150);
      if (lookahead == 'u') ADVANCE(133);
      END_STATE();
    case 80:
      if (lookahead == 'e') ADVANCE(22);
      END_STATE();
    case 81:
      if (lookahead == 'e') ADVANCE(222);
      if (lookahead == 't') ADVANCE(101);
      END_STATE();
    case 82:
      if (lookahead == 'e') ADVANCE(223);
      END_STATE();
    case 83:
      if (lookahead == 'f') ADVANCE(335);
      END_STATE();
    case 84:
      if (lookahead == 'f') ADVANCE(282);
      END_STATE();
    case 85:
      if (lookahead == 'f') ADVANCE(224);
      END_STATE();
    case 86:
      if (lookahead == 'f') ADVANCE(19);
      END_STATE();
    case 87:
      if (lookahead == 'f') ADVANCE(219);
      if (lookahead == 's') ADVANCE(93);
      END_STATE();
    case 88:
      if (lookahead == 'g') ADVANCE(55);
      END_STATE();
    case 89:
      if (lookahead == 'g') ADVANCE(62);
      END_STATE();
    case 90:
      if (lookahead == 'h') ADVANCE(288);
      END_STATE();
    case 91:
      if (lookahead == 'h') ADVANCE(74);
      END_STATE();
    case 92:
      if (lookahead == 'h') ADVANCE(69);
      END_STATE();
    case 93:
      if (lookahead == 'i') ADVANCE(227);
      if (lookahead == 't') ADVANCE(23);
      END_STATE();
    case 94:
      if (lookahead == 'i') ADVANCE(160);
      END_STATE();
    case 95:
      if (lookahead == 'i') ADVANCE(144);
      END_STATE();
    case 96:
      if (lookahead == 'i') ADVANCE(125);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(327);
      END_STATE();
    case 97:
      if (lookahead == 'i') ADVANCE(85);
      END_STATE();
    case 98:
      if (lookahead == 'i') ADVANCE(110);
      if (lookahead == 'l') ADVANCE(176);
      END_STATE();
    case 99:
      if (lookahead == 'i') ADVANCE(37);
      END_STATE();
    case 100:
      if (lookahead == 'i') ADVANCE(137);
      END_STATE();
    case 101:
      if (lookahead == 'i') ADVANCE(111);
      END_STATE();
    case 102:
      if (lookahead == 'i') ADVANCE(27);
      END_STATE();
    case 103:
      if (lookahead == 'i') ADVANCE(145);
      END_STATE();
    case 104:
      if (lookahead == 'i') ADVANCE(200);
      END_STATE();
    case 105:
      if (lookahead == 'i') ADVANCE(146);
      END_STATE();
    case 106:
      if (lookahead == 'i') ADVANCE(147);
      END_STATE();
    case 107:
      if (lookahead == 'i') ADVANCE(202);
      END_STATE();
    case 108:
      if (lookahead == 'k') ADVANCE(280);
      END_STATE();
    case 109:
      if (lookahead == 'l') ADVANCE(35);
      END_STATE();
    case 110:
      if (lookahead == 'l') ADVANCE(300);
      END_STATE();
    case 111:
      if (lookahead == 'l') ADVANCE(272);
      END_STATE();
    case 112:
      if (lookahead == 'l') ADVANCE(208);
      END_STATE();
    case 113:
      if (lookahead == 'l') ADVANCE(25);
      END_STATE();
    case 114:
      if (lookahead == 'l') ADVANCE(97);
      END_STATE();
    case 115:
      if (lookahead == 'l') ADVANCE(51);
      if (lookahead == 'n') ADVANCE(271);
      END_STATE();
    case 116:
      if (lookahead == 'l') ADVANCE(72);
      END_STATE();
    case 117:
      if (lookahead == 'l') ADVANCE(187);
      END_STATE();
    case 118:
      if (lookahead == 'l') ADVANCE(65);
      END_STATE();
    case 119:
      if (lookahead == 'l') ADVANCE(66);
      END_STATE();
    case 120:
      if (lookahead == 'l') ADVANCE(30);
      if (lookahead == 'p') ADVANCE(80);
      if (lookahead == 'w') ADVANCE(163);
      END_STATE();
    case 121:
      if (lookahead == 'm') ADVANCE(149);
      END_STATE();
    case 122:
      if (lookahead == 'm') ADVANCE(53);
      END_STATE();
    case 123:
      if (lookahead == 'm') ADVANCE(78);
      END_STATE();
    case 124:
      if (lookahead == 'n') ADVANCE(39);
      END_STATE();
    case 125:
      if (lookahead == 'n') ADVANCE(83);
      END_STATE();
    case 126:
      if (lookahead == 'n') ADVANCE(9);
      END_STATE();
    case 127:
      if (lookahead == 'n') ADVANCE(81);
      END_STATE();
    case 128:
      if (lookahead == 'n') ADVANCE(264);
      END_STATE();
    case 129:
      if (lookahead == 'n') ADVANCE(311);
      END_STATE();
    case 130:
      if (lookahead == 'n') ADVANCE(246);
      END_STATE();
    case 131:
      if (lookahead == 'n') ADVANCE(254);
      END_STATE();
    case 132:
      if (lookahead == 'n') ADVANCE(292);
      END_STATE();
    case 133:
      if (lookahead == 'n') ADVANCE(269);
      END_STATE();
    case 134:
      if (lookahead == 'n') ADVANCE(95);
      END_STATE();
    case 135:
      if (lookahead == 'n') ADVANCE(40);
      END_STATE();
    case 136:
      if (lookahead == 'n') ADVANCE(99);
      END_STATE();
    case 137:
      if (lookahead == 'n') ADVANCE(182);
      END_STATE();
    case 138:
      if (lookahead == 'n') ADVANCE(196);
      END_STATE();
    case 139:
      if (lookahead == 'n') ADVANCE(194);
      END_STATE();
    case 140:
      if (lookahead == 'n') ADVANCE(45);
      END_STATE();
    case 141:
      if (lookahead == 'o') ADVANCE(175);
      END_STATE();
    case 142:
      if (lookahead == 'o') ADVANCE(143);
      END_STATE();
    case 143:
      if (lookahead == 'o') ADVANCE(84);
      END_STATE();
    case 144:
      if (lookahead == 'o') ADVANCE(129);
      END_STATE();
    case 145:
      if (lookahead == 'o') ADVANCE(130);
      END_STATE();
    case 146:
      if (lookahead == 'o') ADVANCE(131);
      END_STATE();
    case 147:
      if (lookahead == 'o') ADVANCE(132);
      END_STATE();
    case 148:
      if (lookahead == 'p') ADVANCE(290);
      END_STATE();
    case 149:
      if (lookahead == 'p') ADVANCE(114);
      END_STATE();
    case 150:
      if (lookahead == 'p') ADVANCE(80);
      END_STATE();
    case 151:
      if (lookahead == 'p') ADVANCE(158);
      END_STATE();
    case 152:
      if (lookahead == 'p') ADVANCE(60);
      END_STATE();
    case 153:
      if (lookahead == 'p') ADVANCE(215);
      END_STATE();
    case 154:
      if (lookahead == 'q') ADVANCE(305);
      END_STATE();
    case 155:
      if (lookahead == 'q') ADVANCE(305);
      if (lookahead == 't') ADVANCE(308);
      END_STATE();
    case 156:
      if (lookahead == 'r') ADVANCE(88);
      END_STATE();
    case 157:
      if (lookahead == 'r') ADVANCE(225);
      END_STATE();
    case 158:
      if (lookahead == 'r') ADVANCE(142);
      END_STATE();
    case 159:
      if (lookahead == 'r') ADVANCE(102);
      END_STATE();
    case 160:
      if (lookahead == 'r') ADVANCE(50);
      END_STATE();
    case 161:
      if (lookahead == 'r') ADVANCE(179);
      END_STATE();
    case 162:
      if (lookahead == 'r') ADVANCE(26);
      END_STATE();
    case 163:
      if (lookahead == 'r') ADVANCE(104);
      END_STATE();
    case 164:
      if (lookahead == 'r') ADVANCE(20);
      END_STATE();
    case 165:
      if (lookahead == 'r') ADVANCE(57);
      END_STATE();
    case 166:
      if (lookahead == 'r') ADVANCE(89);
      END_STATE();
    case 167:
      if (lookahead == 'r') ADVANCE(213);
      END_STATE();
    case 168:
      if (lookahead == 'r') ADVANCE(29);
      END_STATE();
    case 169:
      if (lookahead == 'r') ADVANCE(28);
      END_STATE();
    case 170:
      if (lookahead == 'r') ADVANCE(107);
      END_STATE();
    case 171:
      if (lookahead == 's') ADVANCE(279);
      END_STATE();
    case 172:
      if (lookahead == 's') ADVANCE(286);
      END_STATE();
    case 173:
      if (lookahead == 's') ADVANCE(90);
      END_STATE();
    case 174:
      if (lookahead == 's') ADVANCE(43);
      END_STATE();
    case 175:
      if (lookahead == 's') ADVANCE(180);
      END_STATE();
    case 176:
      if (lookahead == 's') ADVANCE(54);
      END_STATE();
    case 177:
      if (lookahead == 's') ADVANCE(77);
      END_STATE();
    case 178:
      if (lookahead == 't') ADVANCE(267);
      END_STATE();
    case 179:
      if (lookahead == 't') ADVANCE(244);
      END_STATE();
    case 180:
      if (lookahead == 't') ADVANCE(307);
      END_STATE();
    case 181:
      if (lookahead == 't') ADVANCE(296);
      END_STATE();
    case 182:
      if (lookahead == 't') ADVANCE(3);
      END_STATE();
    case 183:
      if (lookahead == 't') ADVANCE(298);
      END_STATE();
    case 184:
      if (lookahead == 't') ADVANCE(306);
      END_STATE();
    case 185:
      if (lookahead == 't') ADVANCE(315);
      END_STATE();
    case 186:
      if (lookahead == 't') ADVANCE(256);
      END_STATE();
    case 187:
      if (lookahead == 't') ADVANCE(251);
      END_STATE();
    case 188:
      if (lookahead == 't') ADVANCE(260);
      END_STATE();
    case 189:
      if (lookahead == 't') ADVANCE(277);
      END_STATE();
    case 190:
      if (lookahead == 't') ADVANCE(153);
      END_STATE();
    case 191:
      if (lookahead == 't') ADVANCE(226);
      END_STATE();
    case 192:
      if (lookahead == 't') ADVANCE(21);
      END_STATE();
    case 193:
      if (lookahead == 't') ADVANCE(211);
      END_STATE();
    case 194:
      if (lookahead == 't') ADVANCE(101);
      END_STATE();
    case 195:
      if (lookahead == 't') ADVANCE(162);
      END_STATE();
    case 196:
      if (lookahead == 't') ADVANCE(171);
      END_STATE();
    case 197:
      if (lookahead == 't') ADVANCE(172);
      END_STATE();
    case 198:
      if (lookahead == 't') ADVANCE(14);
      END_STATE();
    case 199:
      if (lookahead == 't') ADVANCE(56);
      END_STATE();
    case 200:
      if (lookahead == 't') ADVANCE(59);
      END_STATE();
    case 201:
      if (lookahead == 't') ADVANCE(61);
      END_STATE();
    case 202:
      if (lookahead == 't') ADVANCE(63);
      END_STATE();
    case 203:
      if (lookahead == 't') ADVANCE(103);
      END_STATE();
    case 204:
      if (lookahead == 't') ADVANCE(164);
      END_STATE();
    case 205:
      if (lookahead == 't') ADVANCE(105);
      END_STATE();
    case 206:
      if (lookahead == 't') ADVANCE(106);
      END_STATE();
    case 207:
      if (lookahead == 't') ADVANCE(169);
      END_STATE();
    case 208:
      if (lookahead == 'u') ADVANCE(47);
      END_STATE();
    case 209:
      if (lookahead == 'u') ADVANCE(116);
      END_STATE();
    case 210:
      if (lookahead == 'u') ADVANCE(190);
      END_STATE();
    case 211:
      if (lookahead == 'u') ADVANCE(168);
      END_STATE();
    case 212:
      if (lookahead == 'u') ADVANCE(117);
      END_STATE();
    case 213:
      if (lookahead == 'u') ADVANCE(52);
      END_STATE();
    case 214:
      if (lookahead == 'u') ADVANCE(181);
      END_STATE();
    case 215:
      if (lookahead == 'u') ADVANCE(183);
      END_STATE();
    case 216:
      if (lookahead == 'u') ADVANCE(118);
      END_STATE();
    case 217:
      if (lookahead == 'u') ADVANCE(73);
      END_STATE();
    case 218:
      if (lookahead == 'u') ADVANCE(139);
      END_STATE();
    case 219:
      if (lookahead == 'u') ADVANCE(140);
      END_STATE();
    case 220:
      if (lookahead == 'w') ADVANCE(170);
      END_STATE();
    case 221:
      if (lookahead == 'x') ADVANCE(195);
      END_STATE();
    case 222:
      if (lookahead == 'x') ADVANCE(204);
      END_STATE();
    case 223:
      if (lookahead == 'x') ADVANCE(207);
      END_STATE();
    case 224:
      if (lookahead == 'y') ADVANCE(273);
      END_STATE();
    case 225:
      if (lookahead == 'y') ADVANCE(5);
      END_STATE();
    case 226:
      if (lookahead == 'y') ADVANCE(152);
      END_STATE();
    case 227:
      if (lookahead == 'z') ADVANCE(64);
      END_STATE();
    case 228:
      if (lookahead == '(' ||
          lookahead == '[') ADVANCE(238);
      if (lookahead == ';') ADVANCE(236);
      if (lookahead == 'd') ADVANCE(384);
      if (lookahead == 'e') ADVANCE(493);
      if (lookahead == 'l') ADVANCE(381);
      if (lookahead == 'p') ADVANCE(350);
      if (lookahead == 'u') ADVANCE(429);
      if (lookahead == '\t' ||
          lookahead == '\n' ||
          lookahead == '\r' ||
          lookahead == ' ') ADVANCE(237);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '-' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '_' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('a' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 229:
      if (lookahead == '(' ||
          lookahead == '[') ADVANCE(238);
      if (lookahead == ')' ||
          lookahead == ']') ADVANCE(239);
      if (lookahead == ':') ADVANCE(218);
      if (lookahead == ';') ADVANCE(236);
      if (lookahead == '\t' ||
          lookahead == '\n' ||
          lookahead == '\r' ||
          lookahead == ' ') ADVANCE(237);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(328);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '-' ||
          ('/' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '_' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('a' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 230:
      if (lookahead == ')' ||
          lookahead == ']') ADVANCE(239);
      if (lookahead == ':') ADVANCE(32);
      if (lookahead == ';') ADVANCE(236);
      if (lookahead == '\t' ||
          lookahead == '\n' ||
          lookahead == '\r' ||
          lookahead == ' ') ADVANCE(237);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '-' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '_' ||
          lookahead == '|') ADVANCE(322);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('a' <= lookahead && lookahead <= 'z')) ADVANCE(321);
      END_STATE();
    case 231:
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(331);
      END_STATE();
    case 232:
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(332);
      END_STATE();
    case 233:
      if (eof) ADVANCE(235);
      if (lookahead == '"') ADVANCE(1);
      if (lookahead == '(' ||
          lookahead == '[') ADVANCE(238);
      if (lookahead == ')' ||
          lookahead == ']') ADVANCE(239);
      if (lookahead == '-') ADVANCE(402);
      if (lookahead == ':') ADVANCE(46);
      if (lookahead == ';') ADVANCE(236);
      if (lookahead == 'N') ADVANCE(343);
      if (lookahead == 'f') ADVANCE(345);
      if (lookahead == 'i') ADVANCE(428);
      if (lookahead == 't') ADVANCE(455);
      if (lookahead == '\t' ||
          lookahead == '\n' ||
          lookahead == '\r' ||
          lookahead == ' ') ADVANCE(237);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(327);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          ('/' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '_' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('a' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 234:
      if (eof) ADVANCE(235);
      if (lookahead == '(' ||
          lookahead == '[') ADVANCE(238);
      if (lookahead == ')' ||
          lookahead == ']') ADVANCE(239);
      if (lookahead == ':') ADVANCE(34);
      if (lookahead == ';') ADVANCE(236);
      if (lookahead == 'b') ADVANCE(395);
      if (lookahead == 'c') ADVANCE(341);
      if (lookahead == 'd') ADVANCE(342);
      if (lookahead == 'e') ADVANCE(493);
      if (lookahead == 'f') ADVANCE(344);
      if (lookahead == 'i') ADVANCE(421);
      if (lookahead == 'l') ADVANCE(381);
      if (lookahead == 'o') ADVANCE(486);
      if (lookahead == 'p') ADVANCE(349);
      if (lookahead == 'q') ADVANCE(485);
      if (lookahead == 'r') ADVANCE(366);
      if (lookahead == 's') ADVANCE(383);
      if (lookahead == 'u') ADVANCE(429);
      if (lookahead == '\t' ||
          lookahead == '\n' ||
          lookahead == '\r' ||
          lookahead == ' ') ADVANCE(237);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(328);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '-' ||
          ('/' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '_' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('a' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 235:
      ACCEPT_TOKEN(ts_builtin_sym_end);
      END_STATE();
    case 236:
      ACCEPT_TOKEN(sym_comment);
      if (lookahead != 0 &&
          lookahead != '\n') ADVANCE(236);
      END_STATE();
    case 237:
      ACCEPT_TOKEN(sym_ws);
      if (lookahead == '\t' ||
          lookahead == '\n' ||
          lookahead == '\r' ||
          lookahead == ' ') ADVANCE(237);
      END_STATE();
    case 238:
      ACCEPT_TOKEN(sym_lparen);
      END_STATE();
    case 239:
      ACCEPT_TOKEN(sym_rparen);
      END_STATE();
    case 240:
      ACCEPT_TOKEN(anon_sym_COMMA);
      END_STATE();
    case 241:
      ACCEPT_TOKEN(anon_sym_set_DASHoption);
      if (lookahead == '-' ||
          lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('a' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 242:
      ACCEPT_TOKEN(anon_sym_datatype);
      END_STATE();
    case 243:
      ACCEPT_TOKEN(anon_sym_datatype);
      if (lookahead == '-' ||
          lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('a' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 244:
      ACCEPT_TOKEN(anon_sym_sort);
      END_STATE();
    case 245:
      ACCEPT_TOKEN(anon_sym_sort);
      if (lookahead == '-' ||
          lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('a' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 246:
      ACCEPT_TOKEN(anon_sym_function);
      END_STATE();
    case 247:
      ACCEPT_TOKEN(anon_sym_function);
      if (lookahead == '-' ||
          lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('a' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 248:
      ACCEPT_TOKEN(anon_sym_COLONunextractable);
      END_STATE();
    case 249:
      ACCEPT_TOKEN(anon_sym_COLONon_merge);
      END_STATE();
    case 250:
      ACCEPT_TOKEN(anon_sym_COLONmerge);
      END_STATE();
    case 251:
      ACCEPT_TOKEN(anon_sym_COLONdefault);
      END_STATE();
    case 252:
      ACCEPT_TOKEN(anon_sym_declare);
      END_STATE();
    case 253:
      ACCEPT_TOKEN(anon_sym_declare);
      if (lookahead == '-' ||
          lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('a' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 254:
      ACCEPT_TOKEN(anon_sym_relation);
      END_STATE();
    case 255:
      ACCEPT_TOKEN(anon_sym_relation);
      if (lookahead == '-' ||
          lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('a' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 256:
      ACCEPT_TOKEN(anon_sym_ruleset);
      END_STATE();
    case 257:
      ACCEPT_TOKEN(anon_sym_ruleset);
      if (lookahead == '-' ||
          lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('a' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 258:
      ACCEPT_TOKEN(anon_sym_rule);
      if (lookahead == 's') ADVANCE(76);
      END_STATE();
    case 259:
      ACCEPT_TOKEN(anon_sym_rule);
      if (lookahead == 's') ADVANCE(386);
      if (lookahead == '-' ||
          lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('a' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 260:
      ACCEPT_TOKEN(anon_sym_COLONruleset);
      END_STATE();
    case 261:
      ACCEPT_TOKEN(anon_sym_COLONname);
      END_STATE();
    case 262:
      ACCEPT_TOKEN(anon_sym_rewrite);
      END_STATE();
    case 263:
      ACCEPT_TOKEN(anon_sym_rewrite);
      if (lookahead == '-' ||
          lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('a' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 264:
      ACCEPT_TOKEN(anon_sym_COLONwhen);
      END_STATE();
    case 265:
      ACCEPT_TOKEN(anon_sym_birewrite);
      END_STATE();
    case 266:
      ACCEPT_TOKEN(anon_sym_birewrite);
      if (lookahead == '-' ||
          lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('a' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 267:
      ACCEPT_TOKEN(anon_sym_let);
      END_STATE();
    case 268:
      ACCEPT_TOKEN(anon_sym_let);
      if (lookahead == '-' ||
          lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('a' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 269:
      ACCEPT_TOKEN(anon_sym_run);
      END_STATE();
    case 270:
      ACCEPT_TOKEN(anon_sym_run);
      if (lookahead == '-') ADVANCE(460);
      if (lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('a' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 271:
      ACCEPT_TOKEN(anon_sym_run);
      if (lookahead == '-') ADVANCE(174);
      END_STATE();
    case 272:
      ACCEPT_TOKEN(anon_sym_COLONuntil);
      END_STATE();
    case 273:
      ACCEPT_TOKEN(anon_sym_simplify);
      END_STATE();
    case 274:
      ACCEPT_TOKEN(anon_sym_simplify);
      if (lookahead == '-' ||
          lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('a' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 275:
      ACCEPT_TOKEN(anon_sym_calc);
      END_STATE();
    case 276:
      ACCEPT_TOKEN(anon_sym_calc);
      if (lookahead == '-' ||
          lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('a' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 277:
      ACCEPT_TOKEN(anon_sym_query_DASHextract);
      END_STATE();
    case 278:
      ACCEPT_TOKEN(anon_sym_query_DASHextract);
      if (lookahead == '-' ||
          lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('a' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 279:
      ACCEPT_TOKEN(anon_sym_COLONvariants);
      END_STATE();
    case 280:
      ACCEPT_TOKEN(anon_sym_check);
      if (lookahead == '-') ADVANCE(151);
      END_STATE();
    case 281:
      ACCEPT_TOKEN(anon_sym_check);
      if (lookahead == '-') ADVANCE(444);
      if (lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('a' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 282:
      ACCEPT_TOKEN(anon_sym_check_DASHproof);
      END_STATE();
    case 283:
      ACCEPT_TOKEN(anon_sym_check_DASHproof);
      if (lookahead == '-' ||
          lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('a' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 284:
      ACCEPT_TOKEN(anon_sym_run_DASHschedule);
      END_STATE();
    case 285:
      ACCEPT_TOKEN(anon_sym_run_DASHschedule);
      if (lookahead == '-' ||
          lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('a' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 286:
      ACCEPT_TOKEN(anon_sym_print_DASHstats);
      END_STATE();
    case 287:
      ACCEPT_TOKEN(anon_sym_print_DASHstats);
      if (lookahead == '-' ||
          lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('a' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 288:
      ACCEPT_TOKEN(anon_sym_push);
      END_STATE();
    case 289:
      ACCEPT_TOKEN(anon_sym_push);
      if (lookahead == '-' ||
          lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('a' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 290:
      ACCEPT_TOKEN(anon_sym_pop);
      END_STATE();
    case 291:
      ACCEPT_TOKEN(anon_sym_pop);
      if (lookahead == '-' ||
          lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('a' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 292:
      ACCEPT_TOKEN(anon_sym_print_DASHfunction);
      END_STATE();
    case 293:
      ACCEPT_TOKEN(anon_sym_print_DASHfunction);
      if (lookahead == '-' ||
          lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('a' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 294:
      ACCEPT_TOKEN(anon_sym_print_DASHsize);
      END_STATE();
    case 295:
      ACCEPT_TOKEN(anon_sym_print_DASHsize);
      if (lookahead == '-' ||
          lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('a' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 296:
      ACCEPT_TOKEN(anon_sym_input);
      END_STATE();
    case 297:
      ACCEPT_TOKEN(anon_sym_input);
      if (lookahead == '-' ||
          lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('a' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 298:
      ACCEPT_TOKEN(anon_sym_output);
      END_STATE();
    case 299:
      ACCEPT_TOKEN(anon_sym_output);
      if (lookahead == '-' ||
          lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('a' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 300:
      ACCEPT_TOKEN(anon_sym_fail);
      END_STATE();
    case 301:
      ACCEPT_TOKEN(anon_sym_fail);
      if (lookahead == '-' ||
          lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('a' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 302:
      ACCEPT_TOKEN(anon_sym_include);
      END_STATE();
    case 303:
      ACCEPT_TOKEN(anon_sym_include);
      if (lookahead == '-' ||
          lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('a' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 304:
      ACCEPT_TOKEN(anon_sym_saturate);
      END_STATE();
    case 305:
      ACCEPT_TOKEN(anon_sym_seq);
      END_STATE();
    case 306:
      ACCEPT_TOKEN(anon_sym_repeat);
      END_STATE();
    case 307:
      ACCEPT_TOKEN(anon_sym_COLONcost);
      END_STATE();
    case 308:
      ACCEPT_TOKEN(anon_sym_set);
      END_STATE();
    case 309:
      ACCEPT_TOKEN(anon_sym_delete);
      END_STATE();
    case 310:
      ACCEPT_TOKEN(anon_sym_delete);
      if (lookahead == '-' ||
          lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('a' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 311:
      ACCEPT_TOKEN(anon_sym_union);
      END_STATE();
    case 312:
      ACCEPT_TOKEN(anon_sym_union);
      if (lookahead == '-' ||
          lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('a' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 313:
      ACCEPT_TOKEN(anon_sym_panic);
      END_STATE();
    case 314:
      ACCEPT_TOKEN(anon_sym_panic);
      if (lookahead == '-' ||
          lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('a' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 315:
      ACCEPT_TOKEN(anon_sym_extract);
      END_STATE();
    case 316:
      ACCEPT_TOKEN(anon_sym_extract);
      if (lookahead == '-' ||
          lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('a' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 317:
      ACCEPT_TOKEN(anon_sym_LBRACK);
      END_STATE();
    case 318:
      ACCEPT_TOKEN(anon_sym_RBRACK);
      END_STATE();
    case 319:
      ACCEPT_TOKEN(anon_sym_EQ);
      END_STATE();
    case 320:
      ACCEPT_TOKEN(anon_sym_EQ);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '-' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '_' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('a' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 321:
      ACCEPT_TOKEN(sym_type);
      if (lookahead == '-' ||
          lookahead == '_') ADVANCE(321);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(321);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(322);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('a' <= lookahead && lookahead <= 'z')) ADVANCE(321);
      END_STATE();
    case 322:
      ACCEPT_TOKEN(sym_type);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '-' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '_' ||
          lookahead == '|') ADVANCE(322);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('a' <= lookahead && lookahead <= 'z')) ADVANCE(321);
      END_STATE();
    case 323:
      ACCEPT_TOKEN(anon_sym_true);
      END_STATE();
    case 324:
      ACCEPT_TOKEN(anon_sym_true);
      if (lookahead == '-' ||
          lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('a' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 325:
      ACCEPT_TOKEN(anon_sym_false);
      END_STATE();
    case 326:
      ACCEPT_TOKEN(anon_sym_false);
      if (lookahead == '-' ||
          lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('a' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 327:
      ACCEPT_TOKEN(sym_num);
      if (lookahead == '.') ADVANCE(231);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(327);
      END_STATE();
    case 328:
      ACCEPT_TOKEN(sym_unum);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(328);
      END_STATE();
    case 329:
      ACCEPT_TOKEN(anon_sym_NaN);
      END_STATE();
    case 330:
      ACCEPT_TOKEN(anon_sym_NaN);
      if (lookahead == '-' ||
          lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('a' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 331:
      ACCEPT_TOKEN(aux_sym_f64_token1);
      if (lookahead == 'e') ADVANCE(2);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(331);
      END_STATE();
    case 332:
      ACCEPT_TOKEN(aux_sym_f64_token1);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(332);
      END_STATE();
    case 333:
      ACCEPT_TOKEN(anon_sym_inf);
      END_STATE();
    case 334:
      ACCEPT_TOKEN(anon_sym_inf);
      if (lookahead == '-' ||
          lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('a' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 335:
      ACCEPT_TOKEN(anon_sym_DASHinf);
      END_STATE();
    case 336:
      ACCEPT_TOKEN(anon_sym_DASHinf);
      if (lookahead == '-' ||
          lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('a' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 337:
      ACCEPT_TOKEN(sym_ident);
      if (lookahead == '-') ADVANCE(392);
      if (lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('a' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 338:
      ACCEPT_TOKEN(sym_ident);
      if (lookahead == '-') ADVANCE(435);
      if (lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('a' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 339:
      ACCEPT_TOKEN(sym_ident);
      if (lookahead == '-') ADVANCE(387);
      if (lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('a' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 340:
      ACCEPT_TOKEN(sym_ident);
      if (lookahead == 'N') ADVANCE(330);
      if (lookahead == '-' ||
          lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('a' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 341:
      ACCEPT_TOKEN(sym_ident);
      if (lookahead == 'a') ADVANCE(410);
      if (lookahead == 'h') ADVANCE(379);
      if (lookahead == '-' ||
          lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('b' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 342:
      ACCEPT_TOKEN(sym_ident);
      if (lookahead == 'a') ADVANCE(462);
      if (lookahead == 'e') ADVANCE(357);
      if (lookahead == '-' ||
          lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('b' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 343:
      ACCEPT_TOKEN(sym_ident);
      if (lookahead == 'a') ADVANCE(340);
      if (lookahead == '-' ||
          lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('b' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 344:
      ACCEPT_TOKEN(sym_ident);
      if (lookahead == 'a') ADVANCE(399);
      if (lookahead == 'u') ADVANCE(427);
      if (lookahead == '-' ||
          lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('b' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 345:
      ACCEPT_TOKEN(sym_ident);
      if (lookahead == 'a') ADVANCE(414);
      if (lookahead == '-' ||
          lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('b' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 346:
      ACCEPT_TOKEN(sym_ident);
      if (lookahead == 'a') ADVANCE(473);
      if (lookahead == '-' ||
          lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('b' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 347:
      ACCEPT_TOKEN(sym_ident);
      if (lookahead == 'a') ADVANCE(475);
      if (lookahead == '-' ||
          lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('b' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 348:
      ACCEPT_TOKEN(sym_ident);
      if (lookahead == 'a') ADVANCE(454);
      if (lookahead == '-' ||
          lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('b' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 349:
      ACCEPT_TOKEN(sym_ident);
      if (lookahead == 'a') ADVANCE(431);
      if (lookahead == 'o') ADVANCE(442);
      if (lookahead == 'r') ADVANCE(400);
      if (lookahead == 'u') ADVANCE(458);
      if (lookahead == '-' ||
          lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('b' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 350:
      ACCEPT_TOKEN(sym_ident);
      if (lookahead == 'a') ADVANCE(431);
      if (lookahead == '-' ||
          lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('b' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 351:
      ACCEPT_TOKEN(sym_ident);
      if (lookahead == 'a') ADVANCE(361);
      if (lookahead == '-' ||
          lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('b' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 352:
      ACCEPT_TOKEN(sym_ident);
      if (lookahead == 'a') ADVANCE(362);
      if (lookahead == '-' ||
          lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('b' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 353:
      ACCEPT_TOKEN(sym_ident);
      if (lookahead == 'a') ADVANCE(480);
      if (lookahead == '-' ||
          lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('b' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 354:
      ACCEPT_TOKEN(sym_ident);
      if (lookahead == 'c') ADVANCE(276);
      if (lookahead == '-' ||
          lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('a' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 355:
      ACCEPT_TOKEN(sym_ident);
      if (lookahead == 'c') ADVANCE(409);
      if (lookahead == '-' ||
          lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('a' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 356:
      ACCEPT_TOKEN(sym_ident);
      if (lookahead == 'c') ADVANCE(314);
      if (lookahead == '-' ||
          lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('a' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 357:
      ACCEPT_TOKEN(sym_ident);
      if (lookahead == 'c') ADVANCE(412);
      if (lookahead == 'l') ADVANCE(385);
      if (lookahead == '-' ||
          lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('a' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 358:
      ACCEPT_TOKEN(sym_ident);
      if (lookahead == 'c') ADVANCE(394);
      if (lookahead == '-' ||
          lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('a' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 359:
      ACCEPT_TOKEN(sym_ident);
      if (lookahead == 'c') ADVANCE(413);
      if (lookahead == 'p') ADVANCE(488);
      if (lookahead == '-' ||
          lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('a' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 360:
      ACCEPT_TOKEN(sym_ident);
      if (lookahead == 'c') ADVANCE(479);
      if (lookahead == '-' ||
          lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('a' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 361:
      ACCEPT_TOKEN(sym_ident);
      if (lookahead == 'c') ADVANCE(468);
      if (lookahead == '-' ||
          lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('a' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 362:
      ACCEPT_TOKEN(sym_ident);
      if (lookahead == 'c') ADVANCE(470);
      if (lookahead == '-' ||
          lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('a' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 363:
      ACCEPT_TOKEN(sym_ident);
      if (lookahead == 'c') ADVANCE(482);
      if (lookahead == '-' ||
          lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('a' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 364:
      ACCEPT_TOKEN(sym_ident);
      if (lookahead == 'd') ADVANCE(487);
      if (lookahead == '-' ||
          lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('a' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 365:
      ACCEPT_TOKEN(sym_ident);
      if (lookahead == 'd') ADVANCE(371);
      if (lookahead == '-' ||
          lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('a' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 366:
      ACCEPT_TOKEN(sym_ident);
      if (lookahead == 'e') ADVANCE(419);
      if (lookahead == 'u') ADVANCE(415);
      if (lookahead == '-' ||
          lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('a' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 367:
      ACCEPT_TOKEN(sym_ident);
      if (lookahead == 'e') ADVANCE(492);
      if (lookahead == '-' ||
          lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('a' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 368:
      ACCEPT_TOKEN(sym_ident);
      if (lookahead == 'e') ADVANCE(259);
      if (lookahead == '-' ||
          lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('a' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 369:
      ACCEPT_TOKEN(sym_ident);
      if (lookahead == 'e') ADVANCE(310);
      if (lookahead == '-' ||
          lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('a' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 370:
      ACCEPT_TOKEN(sym_ident);
      if (lookahead == 'e') ADVANCE(253);
      if (lookahead == '-' ||
          lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('a' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 371:
      ACCEPT_TOKEN(sym_ident);
      if (lookahead == 'e') ADVANCE(303);
      if (lookahead == '-' ||
          lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('a' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 372:
      ACCEPT_TOKEN(sym_ident);
      if (lookahead == 'e') ADVANCE(263);
      if (lookahead == '-' ||
          lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('a' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 373:
      ACCEPT_TOKEN(sym_ident);
      if (lookahead == 'e') ADVANCE(243);
      if (lookahead == '-' ||
          lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('a' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 374:
      ACCEPT_TOKEN(sym_ident);
      if (lookahead == 'e') ADVANCE(266);
      if (lookahead == '-' ||
          lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('a' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 375:
      ACCEPT_TOKEN(sym_ident);
      if (lookahead == 'e') ADVANCE(295);
      if (lookahead == '-' ||
          lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('a' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 376:
      ACCEPT_TOKEN(sym_ident);
      if (lookahead == 'e') ADVANCE(285);
      if (lookahead == '-' ||
          lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('a' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 377:
      ACCEPT_TOKEN(sym_ident);
      if (lookahead == 'e') ADVANCE(324);
      if (lookahead == '-' ||
          lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('a' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 378:
      ACCEPT_TOKEN(sym_ident);
      if (lookahead == 'e') ADVANCE(326);
      if (lookahead == '-' ||
          lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('a' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 379:
      ACCEPT_TOKEN(sym_ident);
      if (lookahead == 'e') ADVANCE(355);
      if (lookahead == '-' ||
          lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('a' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 380:
      ACCEPT_TOKEN(sym_ident);
      if (lookahead == 'e') ADVANCE(364);
      if (lookahead == '-' ||
          lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('a' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 381:
      ACCEPT_TOKEN(sym_ident);
      if (lookahead == 'e') ADVANCE(463);
      if (lookahead == '-' ||
          lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('a' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 382:
      ACCEPT_TOKEN(sym_ident);
      if (lookahead == 'e') ADVANCE(448);
      if (lookahead == '-' ||
          lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('a' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 383:
      ACCEPT_TOKEN(sym_ident);
      if (lookahead == 'e') ADVANCE(464);
      if (lookahead == 'i') ADVANCE(420);
      if (lookahead == 'o') ADVANCE(453);
      if (lookahead == '-' ||
          lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('a' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 384:
      ACCEPT_TOKEN(sym_ident);
      if (lookahead == 'e') ADVANCE(418);
      if (lookahead == '-' ||
          lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('a' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 385:
      ACCEPT_TOKEN(sym_ident);
      if (lookahead == 'e') ADVANCE(476);
      if (lookahead == '-' ||
          lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('a' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 386:
      ACCEPT_TOKEN(sym_ident);
      if (lookahead == 'e') ADVANCE(469);
      if (lookahead == '-' ||
          lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('a' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 387:
      ACCEPT_TOKEN(sym_ident);
      if (lookahead == 'e') ADVANCE(494);
      if (lookahead == '-' ||
          lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('a' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 388:
      ACCEPT_TOKEN(sym_ident);
      if (lookahead == 'f') ADVANCE(283);
      if (lookahead == '-' ||
          lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('a' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 389:
      ACCEPT_TOKEN(sym_ident);
      if (lookahead == 'f') ADVANCE(334);
      if (lookahead == '-' ||
          lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('a' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 390:
      ACCEPT_TOKEN(sym_ident);
      if (lookahead == 'f') ADVANCE(336);
      if (lookahead == '-' ||
          lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('a' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 391:
      ACCEPT_TOKEN(sym_ident);
      if (lookahead == 'f') ADVANCE(495);
      if (lookahead == '-' ||
          lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('a' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 392:
      ACCEPT_TOKEN(sym_ident);
      if (lookahead == 'f') ADVANCE(491);
      if (lookahead == 's') ADVANCE(398);
      if (lookahead == '-' ||
          lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('a' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 393:
      ACCEPT_TOKEN(sym_ident);
      if (lookahead == 'h') ADVANCE(289);
      if (lookahead == '-' ||
          lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('a' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 394:
      ACCEPT_TOKEN(sym_ident);
      if (lookahead == 'h') ADVANCE(380);
      if (lookahead == '-' ||
          lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('a' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 395:
      ACCEPT_TOKEN(sym_ident);
      if (lookahead == 'i') ADVANCE(451);
      if (lookahead == '-' ||
          lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('a' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 396:
      ACCEPT_TOKEN(sym_ident);
      if (lookahead == 'i') ADVANCE(437);
      if (lookahead == '-' ||
          lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('a' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 397:
      ACCEPT_TOKEN(sym_ident);
      if (lookahead == 'i') ADVANCE(391);
      if (lookahead == '-' ||
          lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('a' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 398:
      ACCEPT_TOKEN(sym_ident);
      if (lookahead == 'i') ADVANCE(498);
      if (lookahead == 't') ADVANCE(347);
      if (lookahead == '-' ||
          lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('a' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 399:
      ACCEPT_TOKEN(sym_ident);
      if (lookahead == 'i') ADVANCE(411);
      if (lookahead == '-' ||
          lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('a' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 400:
      ACCEPT_TOKEN(sym_ident);
      if (lookahead == 'i') ADVANCE(432);
      if (lookahead == '-' ||
          lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('a' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 401:
      ACCEPT_TOKEN(sym_ident);
      if (lookahead == 'i') ADVANCE(356);
      if (lookahead == '-' ||
          lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('a' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 402:
      ACCEPT_TOKEN(sym_ident);
      if (lookahead == 'i') ADVANCE(430);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(327);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '-' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '_' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('a' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 403:
      ACCEPT_TOKEN(sym_ident);
      if (lookahead == 'i') ADVANCE(477);
      if (lookahead == '-' ||
          lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('a' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 404:
      ACCEPT_TOKEN(sym_ident);
      if (lookahead == 'i') ADVANCE(478);
      if (lookahead == '-' ||
          lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('a' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 405:
      ACCEPT_TOKEN(sym_ident);
      if (lookahead == 'i') ADVANCE(438);
      if (lookahead == '-' ||
          lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('a' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 406:
      ACCEPT_TOKEN(sym_ident);
      if (lookahead == 'i') ADVANCE(439);
      if (lookahead == '-' ||
          lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('a' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 407:
      ACCEPT_TOKEN(sym_ident);
      if (lookahead == 'i') ADVANCE(440);
      if (lookahead == '-' ||
          lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('a' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 408:
      ACCEPT_TOKEN(sym_ident);
      if (lookahead == 'i') ADVANCE(441);
      if (lookahead == '-' ||
          lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('a' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 409:
      ACCEPT_TOKEN(sym_ident);
      if (lookahead == 'k') ADVANCE(281);
      if (lookahead == '-' ||
          lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('a' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 410:
      ACCEPT_TOKEN(sym_ident);
      if (lookahead == 'l') ADVANCE(354);
      if (lookahead == '-' ||
          lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('a' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 411:
      ACCEPT_TOKEN(sym_ident);
      if (lookahead == 'l') ADVANCE(301);
      if (lookahead == '-' ||
          lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('a' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 412:
      ACCEPT_TOKEN(sym_ident);
      if (lookahead == 'l') ADVANCE(348);
      if (lookahead == '-' ||
          lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('a' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 413:
      ACCEPT_TOKEN(sym_ident);
      if (lookahead == 'l') ADVANCE(484);
      if (lookahead == '-' ||
          lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('a' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 414:
      ACCEPT_TOKEN(sym_ident);
      if (lookahead == 'l') ADVANCE(461);
      if (lookahead == '-' ||
          lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('a' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 415:
      ACCEPT_TOKEN(sym_ident);
      if (lookahead == 'l') ADVANCE(368);
      if (lookahead == 'n') ADVANCE(270);
      if (lookahead == '-' ||
          lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('a' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 416:
      ACCEPT_TOKEN(sym_ident);
      if (lookahead == 'l') ADVANCE(397);
      if (lookahead == '-' ||
          lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('a' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 417:
      ACCEPT_TOKEN(sym_ident);
      if (lookahead == 'l') ADVANCE(376);
      if (lookahead == '-' ||
          lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('a' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 418:
      ACCEPT_TOKEN(sym_ident);
      if (lookahead == 'l') ADVANCE(385);
      if (lookahead == '-' ||
          lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('a' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 419:
      ACCEPT_TOKEN(sym_ident);
      if (lookahead == 'l') ADVANCE(353);
      if (lookahead == 'w') ADVANCE(452);
      if (lookahead == '-' ||
          lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('a' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 420:
      ACCEPT_TOKEN(sym_ident);
      if (lookahead == 'm') ADVANCE(443);
      if (lookahead == '-' ||
          lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('a' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 421:
      ACCEPT_TOKEN(sym_ident);
      if (lookahead == 'n') ADVANCE(359);
      if (lookahead == '-' ||
          lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('a' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 422:
      ACCEPT_TOKEN(sym_ident);
      if (lookahead == 'n') ADVANCE(312);
      if (lookahead == '-' ||
          lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('a' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 423:
      ACCEPT_TOKEN(sym_ident);
      if (lookahead == 'n') ADVANCE(247);
      if (lookahead == '-' ||
          lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('a' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 424:
      ACCEPT_TOKEN(sym_ident);
      if (lookahead == 'n') ADVANCE(255);
      if (lookahead == '-' ||
          lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('a' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 425:
      ACCEPT_TOKEN(sym_ident);
      if (lookahead == 'n') ADVANCE(241);
      if (lookahead == '-' ||
          lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('a' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 426:
      ACCEPT_TOKEN(sym_ident);
      if (lookahead == 'n') ADVANCE(293);
      if (lookahead == '-' ||
          lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('a' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 427:
      ACCEPT_TOKEN(sym_ident);
      if (lookahead == 'n') ADVANCE(360);
      if (lookahead == '-' ||
          lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('a' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 428:
      ACCEPT_TOKEN(sym_ident);
      if (lookahead == 'n') ADVANCE(389);
      if (lookahead == '-' ||
          lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('a' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 429:
      ACCEPT_TOKEN(sym_ident);
      if (lookahead == 'n') ADVANCE(396);
      if (lookahead == '-' ||
          lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('a' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 430:
      ACCEPT_TOKEN(sym_ident);
      if (lookahead == 'n') ADVANCE(390);
      if (lookahead == '-' ||
          lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('a' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 431:
      ACCEPT_TOKEN(sym_ident);
      if (lookahead == 'n') ADVANCE(401);
      if (lookahead == '-' ||
          lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('a' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 432:
      ACCEPT_TOKEN(sym_ident);
      if (lookahead == 'n') ADVANCE(472);
      if (lookahead == '-' ||
          lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('a' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 433:
      ACCEPT_TOKEN(sym_ident);
      if (lookahead == 'n') ADVANCE(363);
      if (lookahead == '-' ||
          lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('a' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 434:
      ACCEPT_TOKEN(sym_ident);
      if (lookahead == 'o') ADVANCE(388);
      if (lookahead == '-' ||
          lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('a' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 435:
      ACCEPT_TOKEN(sym_ident);
      if (lookahead == 'o') ADVANCE(447);
      if (lookahead == '-' ||
          lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('a' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 436:
      ACCEPT_TOKEN(sym_ident);
      if (lookahead == 'o') ADVANCE(434);
      if (lookahead == '-' ||
          lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('a' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 437:
      ACCEPT_TOKEN(sym_ident);
      if (lookahead == 'o') ADVANCE(422);
      if (lookahead == '-' ||
          lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('a' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 438:
      ACCEPT_TOKEN(sym_ident);
      if (lookahead == 'o') ADVANCE(423);
      if (lookahead == '-' ||
          lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('a' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 439:
      ACCEPT_TOKEN(sym_ident);
      if (lookahead == 'o') ADVANCE(424);
      if (lookahead == '-' ||
          lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('a' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 440:
      ACCEPT_TOKEN(sym_ident);
      if (lookahead == 'o') ADVANCE(425);
      if (lookahead == '-' ||
          lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('a' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 441:
      ACCEPT_TOKEN(sym_ident);
      if (lookahead == 'o') ADVANCE(426);
      if (lookahead == '-' ||
          lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('a' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 442:
      ACCEPT_TOKEN(sym_ident);
      if (lookahead == 'p') ADVANCE(291);
      if (lookahead == '-' ||
          lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('a' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 443:
      ACCEPT_TOKEN(sym_ident);
      if (lookahead == 'p') ADVANCE(416);
      if (lookahead == '-' ||
          lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('a' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 444:
      ACCEPT_TOKEN(sym_ident);
      if (lookahead == 'p') ADVANCE(450);
      if (lookahead == '-' ||
          lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('a' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 445:
      ACCEPT_TOKEN(sym_ident);
      if (lookahead == 'p') ADVANCE(373);
      if (lookahead == '-' ||
          lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('a' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 446:
      ACCEPT_TOKEN(sym_ident);
      if (lookahead == 'p') ADVANCE(489);
      if (lookahead == '-' ||
          lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('a' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 447:
      ACCEPT_TOKEN(sym_ident);
      if (lookahead == 'p') ADVANCE(481);
      if (lookahead == '-' ||
          lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('a' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 448:
      ACCEPT_TOKEN(sym_ident);
      if (lookahead == 'r') ADVANCE(496);
      if (lookahead == '-' ||
          lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('a' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 449:
      ACCEPT_TOKEN(sym_ident);
      if (lookahead == 'r') ADVANCE(351);
      if (lookahead == '-' ||
          lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('a' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 450:
      ACCEPT_TOKEN(sym_ident);
      if (lookahead == 'r') ADVANCE(436);
      if (lookahead == '-' ||
          lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('a' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 451:
      ACCEPT_TOKEN(sym_ident);
      if (lookahead == 'r') ADVANCE(367);
      if (lookahead == '-' ||
          lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('a' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 452:
      ACCEPT_TOKEN(sym_ident);
      if (lookahead == 'r') ADVANCE(403);
      if (lookahead == '-' ||
          lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('a' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 453:
      ACCEPT_TOKEN(sym_ident);
      if (lookahead == 'r') ADVANCE(465);
      if (lookahead == '-' ||
          lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('a' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 454:
      ACCEPT_TOKEN(sym_ident);
      if (lookahead == 'r') ADVANCE(370);
      if (lookahead == '-' ||
          lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('a' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 455:
      ACCEPT_TOKEN(sym_ident);
      if (lookahead == 'r') ADVANCE(490);
      if (lookahead == '-' ||
          lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('a' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 456:
      ACCEPT_TOKEN(sym_ident);
      if (lookahead == 'r') ADVANCE(404);
      if (lookahead == '-' ||
          lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('a' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 457:
      ACCEPT_TOKEN(sym_ident);
      if (lookahead == 'r') ADVANCE(352);
      if (lookahead == '-' ||
          lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('a' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 458:
      ACCEPT_TOKEN(sym_ident);
      if (lookahead == 's') ADVANCE(393);
      if (lookahead == '-' ||
          lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('a' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 459:
      ACCEPT_TOKEN(sym_ident);
      if (lookahead == 's') ADVANCE(287);
      if (lookahead == '-' ||
          lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('a' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 460:
      ACCEPT_TOKEN(sym_ident);
      if (lookahead == 's') ADVANCE(358);
      if (lookahead == '-' ||
          lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('a' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 461:
      ACCEPT_TOKEN(sym_ident);
      if (lookahead == 's') ADVANCE(378);
      if (lookahead == '-' ||
          lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('a' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 462:
      ACCEPT_TOKEN(sym_ident);
      if (lookahead == 't') ADVANCE(346);
      if (lookahead == '-' ||
          lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('a' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 463:
      ACCEPT_TOKEN(sym_ident);
      if (lookahead == 't') ADVANCE(268);
      if (lookahead == '-' ||
          lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('a' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 464:
      ACCEPT_TOKEN(sym_ident);
      if (lookahead == 't') ADVANCE(338);
      if (lookahead == '-' ||
          lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('a' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 465:
      ACCEPT_TOKEN(sym_ident);
      if (lookahead == 't') ADVANCE(245);
      if (lookahead == '-' ||
          lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('a' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 466:
      ACCEPT_TOKEN(sym_ident);
      if (lookahead == 't') ADVANCE(297);
      if (lookahead == '-' ||
          lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('a' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 467:
      ACCEPT_TOKEN(sym_ident);
      if (lookahead == 't') ADVANCE(299);
      if (lookahead == '-' ||
          lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('a' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 468:
      ACCEPT_TOKEN(sym_ident);
      if (lookahead == 't') ADVANCE(316);
      if (lookahead == '-' ||
          lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('a' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 469:
      ACCEPT_TOKEN(sym_ident);
      if (lookahead == 't') ADVANCE(257);
      if (lookahead == '-' ||
          lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('a' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 470:
      ACCEPT_TOKEN(sym_ident);
      if (lookahead == 't') ADVANCE(278);
      if (lookahead == '-' ||
          lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('a' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 471:
      ACCEPT_TOKEN(sym_ident);
      if (lookahead == 't') ADVANCE(446);
      if (lookahead == '-' ||
          lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('a' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 472:
      ACCEPT_TOKEN(sym_ident);
      if (lookahead == 't') ADVANCE(337);
      if (lookahead == '-' ||
          lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('a' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 473:
      ACCEPT_TOKEN(sym_ident);
      if (lookahead == 't') ADVANCE(497);
      if (lookahead == '-' ||
          lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('a' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 474:
      ACCEPT_TOKEN(sym_ident);
      if (lookahead == 't') ADVANCE(449);
      if (lookahead == '-' ||
          lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('a' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 475:
      ACCEPT_TOKEN(sym_ident);
      if (lookahead == 't') ADVANCE(459);
      if (lookahead == '-' ||
          lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('a' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 476:
      ACCEPT_TOKEN(sym_ident);
      if (lookahead == 't') ADVANCE(369);
      if (lookahead == '-' ||
          lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('a' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 477:
      ACCEPT_TOKEN(sym_ident);
      if (lookahead == 't') ADVANCE(372);
      if (lookahead == '-' ||
          lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('a' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 478:
      ACCEPT_TOKEN(sym_ident);
      if (lookahead == 't') ADVANCE(374);
      if (lookahead == '-' ||
          lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('a' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 479:
      ACCEPT_TOKEN(sym_ident);
      if (lookahead == 't') ADVANCE(405);
      if (lookahead == '-' ||
          lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('a' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 480:
      ACCEPT_TOKEN(sym_ident);
      if (lookahead == 't') ADVANCE(406);
      if (lookahead == '-' ||
          lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('a' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 481:
      ACCEPT_TOKEN(sym_ident);
      if (lookahead == 't') ADVANCE(407);
      if (lookahead == '-' ||
          lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('a' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 482:
      ACCEPT_TOKEN(sym_ident);
      if (lookahead == 't') ADVANCE(408);
      if (lookahead == '-' ||
          lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('a' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 483:
      ACCEPT_TOKEN(sym_ident);
      if (lookahead == 't') ADVANCE(457);
      if (lookahead == '-' ||
          lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('a' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 484:
      ACCEPT_TOKEN(sym_ident);
      if (lookahead == 'u') ADVANCE(365);
      if (lookahead == '-' ||
          lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('a' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 485:
      ACCEPT_TOKEN(sym_ident);
      if (lookahead == 'u') ADVANCE(382);
      if (lookahead == '-' ||
          lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('a' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 486:
      ACCEPT_TOKEN(sym_ident);
      if (lookahead == 'u') ADVANCE(471);
      if (lookahead == '-' ||
          lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('a' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 487:
      ACCEPT_TOKEN(sym_ident);
      if (lookahead == 'u') ADVANCE(417);
      if (lookahead == '-' ||
          lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('a' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 488:
      ACCEPT_TOKEN(sym_ident);
      if (lookahead == 'u') ADVANCE(466);
      if (lookahead == '-' ||
          lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('a' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 489:
      ACCEPT_TOKEN(sym_ident);
      if (lookahead == 'u') ADVANCE(467);
      if (lookahead == '-' ||
          lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('a' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 490:
      ACCEPT_TOKEN(sym_ident);
      if (lookahead == 'u') ADVANCE(377);
      if (lookahead == '-' ||
          lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('a' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 491:
      ACCEPT_TOKEN(sym_ident);
      if (lookahead == 'u') ADVANCE(433);
      if (lookahead == '-' ||
          lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('a' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 492:
      ACCEPT_TOKEN(sym_ident);
      if (lookahead == 'w') ADVANCE(456);
      if (lookahead == '-' ||
          lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('a' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 493:
      ACCEPT_TOKEN(sym_ident);
      if (lookahead == 'x') ADVANCE(474);
      if (lookahead == '-' ||
          lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('a' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 494:
      ACCEPT_TOKEN(sym_ident);
      if (lookahead == 'x') ADVANCE(483);
      if (lookahead == '-' ||
          lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('a' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 495:
      ACCEPT_TOKEN(sym_ident);
      if (lookahead == 'y') ADVANCE(274);
      if (lookahead == '-' ||
          lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('a' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 496:
      ACCEPT_TOKEN(sym_ident);
      if (lookahead == 'y') ADVANCE(339);
      if (lookahead == '-' ||
          lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('a' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 497:
      ACCEPT_TOKEN(sym_ident);
      if (lookahead == 'y') ADVANCE(445);
      if (lookahead == '-' ||
          lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('a' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 498:
      ACCEPT_TOKEN(sym_ident);
      if (lookahead == 'z') ADVANCE(375);
      if (lookahead == '-' ||
          lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('a' <= lookahead && lookahead <= 'y')) ADVANCE(499);
      END_STATE();
    case 499:
      ACCEPT_TOKEN(sym_ident);
      if (lookahead == '-' ||
          lookahead == '_') ADVANCE(499);
      if (('0' <= lookahead && lookahead <= '9')) ADVANCE(499);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('a' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 500:
      ACCEPT_TOKEN(sym_ident);
      if (lookahead == '!' ||
          lookahead == '%' ||
          lookahead == '&' ||
          lookahead == '*' ||
          lookahead == '+' ||
          lookahead == '-' ||
          lookahead == '/' ||
          ('<' <= lookahead && lookahead <= '?') ||
          lookahead == '^' ||
          lookahead == '_' ||
          lookahead == '|') ADVANCE(500);
      if (('A' <= lookahead && lookahead <= 'Z') ||
          ('a' <= lookahead && lookahead <= 'z')) ADVANCE(499);
      END_STATE();
    case 501:
      ACCEPT_TOKEN(sym_string);
      END_STATE();
    default:
      return false;
  }
}

static const TSLexMode ts_lex_modes[STATE_COUNT] = {
  [0] = {.lex_state = 0},
  [1] = {.lex_state = 234},
  [2] = {.lex_state = 234},
  [3] = {.lex_state = 233},
  [4] = {.lex_state = 233},
  [5] = {.lex_state = 233},
  [6] = {.lex_state = 233},
  [7] = {.lex_state = 233},
  [8] = {.lex_state = 233},
  [9] = {.lex_state = 233},
  [10] = {.lex_state = 233},
  [11] = {.lex_state = 233},
  [12] = {.lex_state = 233},
  [13] = {.lex_state = 233},
  [14] = {.lex_state = 233},
  [15] = {.lex_state = 233},
  [16] = {.lex_state = 233},
  [17] = {.lex_state = 233},
  [18] = {.lex_state = 233},
  [19] = {.lex_state = 233},
  [20] = {.lex_state = 233},
  [21] = {.lex_state = 233},
  [22] = {.lex_state = 233},
  [23] = {.lex_state = 233},
  [24] = {.lex_state = 233},
  [25] = {.lex_state = 233},
  [26] = {.lex_state = 233},
  [27] = {.lex_state = 233},
  [28] = {.lex_state = 233},
  [29] = {.lex_state = 233},
  [30] = {.lex_state = 233},
  [31] = {.lex_state = 233},
  [32] = {.lex_state = 233},
  [33] = {.lex_state = 233},
  [34] = {.lex_state = 233},
  [35] = {.lex_state = 233},
  [36] = {.lex_state = 233},
  [37] = {.lex_state = 233},
  [38] = {.lex_state = 233},
  [39] = {.lex_state = 233},
  [40] = {.lex_state = 233},
  [41] = {.lex_state = 233},
  [42] = {.lex_state = 233},
  [43] = {.lex_state = 233},
  [44] = {.lex_state = 233},
  [45] = {.lex_state = 233},
  [46] = {.lex_state = 233},
  [47] = {.lex_state = 233},
  [48] = {.lex_state = 233},
  [49] = {.lex_state = 233},
  [50] = {.lex_state = 233},
  [51] = {.lex_state = 233},
  [52] = {.lex_state = 233},
  [53] = {.lex_state = 233},
  [54] = {.lex_state = 233},
  [55] = {.lex_state = 233},
  [56] = {.lex_state = 233},
  [57] = {.lex_state = 233},
  [58] = {.lex_state = 233},
  [59] = {.lex_state = 233},
  [60] = {.lex_state = 233},
  [61] = {.lex_state = 233},
  [62] = {.lex_state = 228},
  [63] = {.lex_state = 234},
  [64] = {.lex_state = 234},
  [65] = {.lex_state = 234},
  [66] = {.lex_state = 234},
  [67] = {.lex_state = 234},
  [68] = {.lex_state = 234},
  [69] = {.lex_state = 234},
  [70] = {.lex_state = 234},
  [71] = {.lex_state = 234},
  [72] = {.lex_state = 234},
  [73] = {.lex_state = 234},
  [74] = {.lex_state = 234},
  [75] = {.lex_state = 234},
  [76] = {.lex_state = 234},
  [77] = {.lex_state = 234},
  [78] = {.lex_state = 234},
  [79] = {.lex_state = 234},
  [80] = {.lex_state = 229},
  [81] = {.lex_state = 234},
  [82] = {.lex_state = 234},
  [83] = {.lex_state = 234},
  [84] = {.lex_state = 234},
  [85] = {.lex_state = 229},
  [86] = {.lex_state = 234},
  [87] = {.lex_state = 234},
  [88] = {.lex_state = 234},
  [89] = {.lex_state = 229},
  [90] = {.lex_state = 234},
  [91] = {.lex_state = 234},
  [92] = {.lex_state = 234},
  [93] = {.lex_state = 234},
  [94] = {.lex_state = 234},
  [95] = {.lex_state = 234},
  [96] = {.lex_state = 234},
  [97] = {.lex_state = 229},
  [98] = {.lex_state = 229},
  [99] = {.lex_state = 230},
  [100] = {.lex_state = 234},
  [101] = {.lex_state = 229},
  [102] = {.lex_state = 234},
  [103] = {.lex_state = 229},
  [104] = {.lex_state = 229},
  [105] = {.lex_state = 229},
  [106] = {.lex_state = 229},
  [107] = {.lex_state = 230},
  [108] = {.lex_state = 234},
  [109] = {.lex_state = 234},
  [110] = {.lex_state = 234},
  [111] = {.lex_state = 234},
  [112] = {.lex_state = 229},
  [113] = {.lex_state = 234},
  [114] = {.lex_state = 234},
  [115] = {.lex_state = 7},
  [116] = {.lex_state = 234},
  [117] = {.lex_state = 234},
  [118] = {.lex_state = 234},
  [119] = {.lex_state = 230},
  [120] = {.lex_state = 234},
  [121] = {.lex_state = 7},
  [122] = {.lex_state = 234},
  [123] = {.lex_state = 234},
  [124] = {.lex_state = 234},
  [125] = {.lex_state = 230},
  [126] = {.lex_state = 234},
  [127] = {.lex_state = 229},
  [128] = {.lex_state = 234},
  [129] = {.lex_state = 234},
  [130] = {.lex_state = 230},
  [131] = {.lex_state = 234},
  [132] = {.lex_state = 234},
  [133] = {.lex_state = 234},
  [134] = {.lex_state = 234},
  [135] = {.lex_state = 234},
  [136] = {.lex_state = 229},
  [137] = {.lex_state = 234},
  [138] = {.lex_state = 229},
  [139] = {.lex_state = 234},
  [140] = {.lex_state = 234},
  [141] = {.lex_state = 234},
  [142] = {.lex_state = 234},
  [143] = {.lex_state = 229},
  [144] = {.lex_state = 234},
  [145] = {.lex_state = 234},
  [146] = {.lex_state = 234},
  [147] = {.lex_state = 234},
  [148] = {.lex_state = 229},
  [149] = {.lex_state = 234},
  [150] = {.lex_state = 229},
  [151] = {.lex_state = 234},
  [152] = {.lex_state = 234},
  [153] = {.lex_state = 234},
  [154] = {.lex_state = 234},
  [155] = {.lex_state = 234},
  [156] = {.lex_state = 229},
  [157] = {.lex_state = 229},
  [158] = {.lex_state = 234},
  [159] = {.lex_state = 234},
  [160] = {.lex_state = 234},
  [161] = {.lex_state = 234},
  [162] = {.lex_state = 234},
  [163] = {.lex_state = 230},
  [164] = {.lex_state = 230},
  [165] = {.lex_state = 234},
  [166] = {.lex_state = 234},
  [167] = {.lex_state = 234},
  [168] = {.lex_state = 234},
  [169] = {.lex_state = 229},
  [170] = {.lex_state = 234},
  [171] = {.lex_state = 234},
  [172] = {.lex_state = 234},
  [173] = {.lex_state = 234},
  [174] = {.lex_state = 234},
  [175] = {.lex_state = 234},
  [176] = {.lex_state = 234},
  [177] = {.lex_state = 234},
  [178] = {.lex_state = 234},
  [179] = {.lex_state = 234},
  [180] = {.lex_state = 234},
  [181] = {.lex_state = 234},
  [182] = {.lex_state = 234},
  [183] = {.lex_state = 234},
  [184] = {.lex_state = 234},
  [185] = {.lex_state = 234},
  [186] = {.lex_state = 229},
  [187] = {.lex_state = 234},
  [188] = {.lex_state = 234},
  [189] = {.lex_state = 229},
  [190] = {.lex_state = 234},
  [191] = {.lex_state = 234},
  [192] = {.lex_state = 234},
  [193] = {.lex_state = 6},
  [194] = {.lex_state = 234},
  [195] = {.lex_state = 234},
  [196] = {.lex_state = 230},
  [197] = {.lex_state = 229},
  [198] = {.lex_state = 234},
  [199] = {.lex_state = 234},
  [200] = {.lex_state = 229},
  [201] = {.lex_state = 229},
  [202] = {.lex_state = 0},
  [203] = {.lex_state = 234},
  [204] = {.lex_state = 234},
  [205] = {.lex_state = 229},
  [206] = {.lex_state = 234},
  [207] = {.lex_state = 230},
  [208] = {.lex_state = 234},
  [209] = {.lex_state = 234},
  [210] = {.lex_state = 234},
  [211] = {.lex_state = 234},
  [212] = {.lex_state = 229},
  [213] = {.lex_state = 0},
  [214] = {.lex_state = 229},
  [215] = {.lex_state = 234},
  [216] = {.lex_state = 234},
  [217] = {.lex_state = 0},
  [218] = {.lex_state = 229},
  [219] = {.lex_state = 234},
  [220] = {.lex_state = 234},
  [221] = {.lex_state = 229},
  [222] = {.lex_state = 234},
  [223] = {.lex_state = 234},
  [224] = {.lex_state = 0},
  [225] = {.lex_state = 234},
  [226] = {.lex_state = 229},
  [227] = {.lex_state = 229},
  [228] = {.lex_state = 234},
  [229] = {.lex_state = 234},
  [230] = {.lex_state = 0},
  [231] = {.lex_state = 229},
  [232] = {.lex_state = 229},
  [233] = {.lex_state = 0},
  [234] = {.lex_state = 234},
  [235] = {.lex_state = 0},
  [236] = {.lex_state = 234},
  [237] = {.lex_state = 230},
  [238] = {.lex_state = 0},
  [239] = {.lex_state = 229},
  [240] = {.lex_state = 234},
  [241] = {.lex_state = 234},
  [242] = {.lex_state = 0},
  [243] = {.lex_state = 234},
  [244] = {.lex_state = 234},
  [245] = {.lex_state = 0},
  [246] = {.lex_state = 234},
  [247] = {.lex_state = 234},
  [248] = {.lex_state = 234},
  [249] = {.lex_state = 229},
  [250] = {.lex_state = 234},
  [251] = {.lex_state = 234},
  [252] = {.lex_state = 234},
  [253] = {.lex_state = 234},
  [254] = {.lex_state = 234},
  [255] = {.lex_state = 230},
  [256] = {.lex_state = 229},
  [257] = {.lex_state = 229},
  [258] = {.lex_state = 229},
  [259] = {.lex_state = 0},
  [260] = {.lex_state = 229},
  [261] = {.lex_state = 229},
  [262] = {.lex_state = 234},
  [263] = {.lex_state = 229},
  [264] = {.lex_state = 234},
  [265] = {.lex_state = 234},
  [266] = {.lex_state = 234},
  [267] = {.lex_state = 234},
  [268] = {.lex_state = 229},
  [269] = {.lex_state = 234},
};

static const uint16_t ts_parse_table[LARGE_STATE_COUNT][SYMBOL_COUNT] = {
  [0] = {
    [ts_builtin_sym_end] = ACTIONS(1),
    [sym_comment] = ACTIONS(3),
    [sym_ws] = ACTIONS(3),
    [sym_lparen] = ACTIONS(1),
    [sym_rparen] = ACTIONS(1),
    [anon_sym_COMMA] = ACTIONS(1),
    [anon_sym_datatype] = ACTIONS(1),
    [anon_sym_sort] = ACTIONS(1),
    [anon_sym_function] = ACTIONS(1),
    [anon_sym_COLONunextractable] = ACTIONS(1),
    [anon_sym_COLONon_merge] = ACTIONS(1),
    [anon_sym_COLONmerge] = ACTIONS(1),
    [anon_sym_COLONdefault] = ACTIONS(1),
    [anon_sym_declare] = ACTIONS(1),
    [anon_sym_relation] = ACTIONS(1),
    [anon_sym_ruleset] = ACTIONS(1),
    [anon_sym_rule] = ACTIONS(1),
    [anon_sym_COLONruleset] = ACTIONS(1),
    [anon_sym_COLONname] = ACTIONS(1),
    [anon_sym_rewrite] = ACTIONS(1),
    [anon_sym_COLONwhen] = ACTIONS(1),
    [anon_sym_birewrite] = ACTIONS(1),
    [anon_sym_let] = ACTIONS(1),
    [anon_sym_run] = ACTIONS(1),
    [anon_sym_COLONuntil] = ACTIONS(1),
    [anon_sym_simplify] = ACTIONS(1),
    [anon_sym_calc] = ACTIONS(1),
    [anon_sym_query_DASHextract] = ACTIONS(1),
    [anon_sym_COLONvariants] = ACTIONS(1),
    [anon_sym_check] = ACTIONS(1),
    [anon_sym_check_DASHproof] = ACTIONS(1),
    [anon_sym_run_DASHschedule] = ACTIONS(1),
    [anon_sym_print_DASHstats] = ACTIONS(1),
    [anon_sym_push] = ACTIONS(1),
    [anon_sym_pop] = ACTIONS(1),
    [anon_sym_print_DASHfunction] = ACTIONS(1),
    [anon_sym_print_DASHsize] = ACTIONS(1),
    [anon_sym_input] = ACTIONS(1),
    [anon_sym_output] = ACTIONS(1),
    [anon_sym_fail] = ACTIONS(1),
    [anon_sym_include] = ACTIONS(1),
    [anon_sym_saturate] = ACTIONS(1),
    [anon_sym_seq] = ACTIONS(1),
    [anon_sym_repeat] = ACTIONS(1),
    [anon_sym_COLONcost] = ACTIONS(1),
    [anon_sym_set] = ACTIONS(1),
    [anon_sym_delete] = ACTIONS(1),
    [anon_sym_union] = ACTIONS(1),
    [anon_sym_panic] = ACTIONS(1),
    [anon_sym_extract] = ACTIONS(1),
    [anon_sym_LBRACK] = ACTIONS(1),
    [anon_sym_RBRACK] = ACTIONS(1),
    [anon_sym_EQ] = ACTIONS(1),
    [anon_sym_true] = ACTIONS(1),
    [anon_sym_false] = ACTIONS(1),
    [sym_num] = ACTIONS(1),
    [sym_unum] = ACTIONS(1),
    [anon_sym_NaN] = ACTIONS(1),
    [aux_sym_f64_token1] = ACTIONS(1),
    [anon_sym_inf] = ACTIONS(1),
    [anon_sym_DASHinf] = ACTIONS(1),
    [sym_string] = ACTIONS(1),
  },
  [1] = {
    [sym_source_file] = STATE(217),
    [sym_command] = STATE(64),
    [sym_nonletaction] = STATE(155),
    [sym_callexpr] = STATE(131),
    [aux_sym_source_file_repeat1] = STATE(64),
    [ts_builtin_sym_end] = ACTIONS(5),
    [sym_comment] = ACTIONS(3),
    [sym_ws] = ACTIONS(3),
    [sym_lparen] = ACTIONS(7),
  },
};

static const uint16_t ts_small_parse_table[] = {
  [0] = 30,
    ACTIONS(9), 1,
      sym_lparen,
    ACTIONS(13), 1,
      anon_sym_datatype,
    ACTIONS(15), 1,
      anon_sym_sort,
    ACTIONS(17), 1,
      anon_sym_function,
    ACTIONS(19), 1,
      anon_sym_declare,
    ACTIONS(21), 1,
      anon_sym_relation,
    ACTIONS(23), 1,
      anon_sym_ruleset,
    ACTIONS(25), 1,
      anon_sym_rule,
    ACTIONS(29), 1,
      anon_sym_run,
    ACTIONS(31), 1,
      anon_sym_simplify,
    ACTIONS(33), 1,
      anon_sym_calc,
    ACTIONS(35), 1,
      anon_sym_query_DASHextract,
    ACTIONS(37), 1,
      anon_sym_check,
    ACTIONS(41), 1,
      anon_sym_run_DASHschedule,
    ACTIONS(45), 1,
      anon_sym_print_DASHfunction,
    ACTIONS(47), 1,
      anon_sym_print_DASHsize,
    ACTIONS(49), 1,
      anon_sym_input,
    ACTIONS(51), 1,
      anon_sym_output,
    ACTIONS(53), 1,
      anon_sym_fail,
    ACTIONS(55), 1,
      anon_sym_include,
    ACTIONS(57), 1,
      anon_sym_delete,
    ACTIONS(59), 1,
      anon_sym_union,
    ACTIONS(61), 1,
      anon_sym_panic,
    ACTIONS(63), 1,
      anon_sym_extract,
    ACTIONS(65), 1,
      sym_ident,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
    ACTIONS(11), 2,
      anon_sym_set_DASHoption,
      anon_sym_let,
    ACTIONS(27), 2,
      anon_sym_rewrite,
      anon_sym_birewrite,
    ACTIONS(39), 2,
      anon_sym_check_DASHproof,
      anon_sym_print_DASHstats,
    ACTIONS(43), 2,
      anon_sym_push,
      anon_sym_pop,
  [96] = 12,
    ACTIONS(67), 1,
      sym_lparen,
    ACTIONS(69), 1,
      sym_rparen,
    ACTIONS(73), 1,
      sym_num,
    ACTIONS(77), 1,
      aux_sym_f64_token1,
    ACTIONS(79), 1,
      sym_ident,
    ACTIONS(81), 1,
      sym_string,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
    ACTIONS(71), 2,
      anon_sym_true,
      anon_sym_false,
    STATE(8), 2,
      sym_expr,
      aux_sym_command_repeat2,
    STATE(54), 2,
      sym_literal,
      sym_callexpr,
    ACTIONS(75), 3,
      anon_sym_NaN,
      anon_sym_inf,
      anon_sym_DASHinf,
    STATE(52), 4,
      sym_unit,
      sym_bool,
      sym_f64,
      sym_symstring,
  [142] = 12,
    ACTIONS(67), 1,
      sym_lparen,
    ACTIONS(73), 1,
      sym_num,
    ACTIONS(77), 1,
      aux_sym_f64_token1,
    ACTIONS(79), 1,
      sym_ident,
    ACTIONS(81), 1,
      sym_string,
    ACTIONS(83), 1,
      sym_rparen,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
    ACTIONS(71), 2,
      anon_sym_true,
      anon_sym_false,
    STATE(8), 2,
      sym_expr,
      aux_sym_command_repeat2,
    STATE(54), 2,
      sym_literal,
      sym_callexpr,
    ACTIONS(75), 3,
      anon_sym_NaN,
      anon_sym_inf,
      anon_sym_DASHinf,
    STATE(52), 4,
      sym_unit,
      sym_bool,
      sym_f64,
      sym_symstring,
  [188] = 12,
    ACTIONS(67), 1,
      sym_lparen,
    ACTIONS(73), 1,
      sym_num,
    ACTIONS(77), 1,
      aux_sym_f64_token1,
    ACTIONS(79), 1,
      sym_ident,
    ACTIONS(81), 1,
      sym_string,
    ACTIONS(85), 1,
      sym_rparen,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
    ACTIONS(71), 2,
      anon_sym_true,
      anon_sym_false,
    STATE(8), 2,
      sym_expr,
      aux_sym_command_repeat2,
    STATE(54), 2,
      sym_literal,
      sym_callexpr,
    ACTIONS(75), 3,
      anon_sym_NaN,
      anon_sym_inf,
      anon_sym_DASHinf,
    STATE(52), 4,
      sym_unit,
      sym_bool,
      sym_f64,
      sym_symstring,
  [234] = 12,
    ACTIONS(67), 1,
      sym_lparen,
    ACTIONS(73), 1,
      sym_num,
    ACTIONS(77), 1,
      aux_sym_f64_token1,
    ACTIONS(79), 1,
      sym_ident,
    ACTIONS(81), 1,
      sym_string,
    ACTIONS(87), 1,
      sym_rparen,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
    ACTIONS(71), 2,
      anon_sym_true,
      anon_sym_false,
    STATE(8), 2,
      sym_expr,
      aux_sym_command_repeat2,
    STATE(54), 2,
      sym_literal,
      sym_callexpr,
    ACTIONS(75), 3,
      anon_sym_NaN,
      anon_sym_inf,
      anon_sym_DASHinf,
    STATE(52), 4,
      sym_unit,
      sym_bool,
      sym_f64,
      sym_symstring,
  [280] = 12,
    ACTIONS(67), 1,
      sym_lparen,
    ACTIONS(73), 1,
      sym_num,
    ACTIONS(77), 1,
      aux_sym_f64_token1,
    ACTIONS(79), 1,
      sym_ident,
    ACTIONS(81), 1,
      sym_string,
    ACTIONS(89), 1,
      sym_rparen,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
    ACTIONS(71), 2,
      anon_sym_true,
      anon_sym_false,
    STATE(4), 2,
      sym_expr,
      aux_sym_command_repeat2,
    STATE(54), 2,
      sym_literal,
      sym_callexpr,
    ACTIONS(75), 3,
      anon_sym_NaN,
      anon_sym_inf,
      anon_sym_DASHinf,
    STATE(52), 4,
      sym_unit,
      sym_bool,
      sym_f64,
      sym_symstring,
  [326] = 12,
    ACTIONS(91), 1,
      sym_lparen,
    ACTIONS(94), 1,
      sym_rparen,
    ACTIONS(99), 1,
      sym_num,
    ACTIONS(105), 1,
      aux_sym_f64_token1,
    ACTIONS(108), 1,
      sym_ident,
    ACTIONS(111), 1,
      sym_string,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
    ACTIONS(96), 2,
      anon_sym_true,
      anon_sym_false,
    STATE(8), 2,
      sym_expr,
      aux_sym_command_repeat2,
    STATE(54), 2,
      sym_literal,
      sym_callexpr,
    ACTIONS(102), 3,
      anon_sym_NaN,
      anon_sym_inf,
      anon_sym_DASHinf,
    STATE(52), 4,
      sym_unit,
      sym_bool,
      sym_f64,
      sym_symstring,
  [372] = 12,
    ACTIONS(67), 1,
      sym_lparen,
    ACTIONS(73), 1,
      sym_num,
    ACTIONS(77), 1,
      aux_sym_f64_token1,
    ACTIONS(79), 1,
      sym_ident,
    ACTIONS(81), 1,
      sym_string,
    ACTIONS(114), 1,
      sym_rparen,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
    ACTIONS(71), 2,
      anon_sym_true,
      anon_sym_false,
    STATE(5), 2,
      sym_expr,
      aux_sym_command_repeat2,
    STATE(54), 2,
      sym_literal,
      sym_callexpr,
    ACTIONS(75), 3,
      anon_sym_NaN,
      anon_sym_inf,
      anon_sym_DASHinf,
    STATE(52), 4,
      sym_unit,
      sym_bool,
      sym_f64,
      sym_symstring,
  [418] = 12,
    ACTIONS(67), 1,
      sym_lparen,
    ACTIONS(73), 1,
      sym_num,
    ACTIONS(77), 1,
      aux_sym_f64_token1,
    ACTIONS(79), 1,
      sym_ident,
    ACTIONS(81), 1,
      sym_string,
    ACTIONS(116), 1,
      sym_rparen,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
    ACTIONS(71), 2,
      anon_sym_true,
      anon_sym_false,
    STATE(8), 2,
      sym_expr,
      aux_sym_command_repeat2,
    STATE(54), 2,
      sym_literal,
      sym_callexpr,
    ACTIONS(75), 3,
      anon_sym_NaN,
      anon_sym_inf,
      anon_sym_DASHinf,
    STATE(52), 4,
      sym_unit,
      sym_bool,
      sym_f64,
      sym_symstring,
  [464] = 12,
    ACTIONS(67), 1,
      sym_lparen,
    ACTIONS(73), 1,
      sym_num,
    ACTIONS(77), 1,
      aux_sym_f64_token1,
    ACTIONS(79), 1,
      sym_ident,
    ACTIONS(81), 1,
      sym_string,
    ACTIONS(118), 1,
      sym_rparen,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
    ACTIONS(71), 2,
      anon_sym_true,
      anon_sym_false,
    STATE(14), 2,
      sym_expr,
      aux_sym_command_repeat2,
    STATE(54), 2,
      sym_literal,
      sym_callexpr,
    ACTIONS(75), 3,
      anon_sym_NaN,
      anon_sym_inf,
      anon_sym_DASHinf,
    STATE(52), 4,
      sym_unit,
      sym_bool,
      sym_f64,
      sym_symstring,
  [510] = 12,
    ACTIONS(67), 1,
      sym_lparen,
    ACTIONS(73), 1,
      sym_num,
    ACTIONS(77), 1,
      aux_sym_f64_token1,
    ACTIONS(79), 1,
      sym_ident,
    ACTIONS(81), 1,
      sym_string,
    ACTIONS(120), 1,
      sym_rparen,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
    ACTIONS(71), 2,
      anon_sym_true,
      anon_sym_false,
    STATE(3), 2,
      sym_expr,
      aux_sym_command_repeat2,
    STATE(54), 2,
      sym_literal,
      sym_callexpr,
    ACTIONS(75), 3,
      anon_sym_NaN,
      anon_sym_inf,
      anon_sym_DASHinf,
    STATE(52), 4,
      sym_unit,
      sym_bool,
      sym_f64,
      sym_symstring,
  [556] = 12,
    ACTIONS(67), 1,
      sym_lparen,
    ACTIONS(73), 1,
      sym_num,
    ACTIONS(77), 1,
      aux_sym_f64_token1,
    ACTIONS(79), 1,
      sym_ident,
    ACTIONS(81), 1,
      sym_string,
    ACTIONS(122), 1,
      sym_rparen,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
    ACTIONS(71), 2,
      anon_sym_true,
      anon_sym_false,
    STATE(8), 2,
      sym_expr,
      aux_sym_command_repeat2,
    STATE(54), 2,
      sym_literal,
      sym_callexpr,
    ACTIONS(75), 3,
      anon_sym_NaN,
      anon_sym_inf,
      anon_sym_DASHinf,
    STATE(52), 4,
      sym_unit,
      sym_bool,
      sym_f64,
      sym_symstring,
  [602] = 12,
    ACTIONS(67), 1,
      sym_lparen,
    ACTIONS(73), 1,
      sym_num,
    ACTIONS(77), 1,
      aux_sym_f64_token1,
    ACTIONS(79), 1,
      sym_ident,
    ACTIONS(81), 1,
      sym_string,
    ACTIONS(124), 1,
      sym_rparen,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
    ACTIONS(71), 2,
      anon_sym_true,
      anon_sym_false,
    STATE(8), 2,
      sym_expr,
      aux_sym_command_repeat2,
    STATE(54), 2,
      sym_literal,
      sym_callexpr,
    ACTIONS(75), 3,
      anon_sym_NaN,
      anon_sym_inf,
      anon_sym_DASHinf,
    STATE(52), 4,
      sym_unit,
      sym_bool,
      sym_f64,
      sym_symstring,
  [648] = 12,
    ACTIONS(67), 1,
      sym_lparen,
    ACTIONS(73), 1,
      sym_num,
    ACTIONS(77), 1,
      aux_sym_f64_token1,
    ACTIONS(79), 1,
      sym_ident,
    ACTIONS(81), 1,
      sym_string,
    STATE(8), 1,
      aux_sym_command_repeat2,
    STATE(56), 1,
      sym_expr,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
    ACTIONS(71), 2,
      anon_sym_true,
      anon_sym_false,
    STATE(54), 2,
      sym_literal,
      sym_callexpr,
    ACTIONS(75), 3,
      anon_sym_NaN,
      anon_sym_inf,
      anon_sym_DASHinf,
    STATE(52), 4,
      sym_unit,
      sym_bool,
      sym_f64,
      sym_symstring,
  [693] = 11,
    ACTIONS(67), 1,
      sym_lparen,
    ACTIONS(73), 1,
      sym_num,
    ACTIONS(77), 1,
      aux_sym_f64_token1,
    ACTIONS(79), 1,
      sym_ident,
    ACTIONS(81), 1,
      sym_string,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
    ACTIONS(71), 2,
      anon_sym_true,
      anon_sym_false,
    STATE(13), 2,
      sym_expr,
      aux_sym_command_repeat2,
    STATE(54), 2,
      sym_literal,
      sym_callexpr,
    ACTIONS(75), 3,
      anon_sym_NaN,
      anon_sym_inf,
      anon_sym_DASHinf,
    STATE(52), 4,
      sym_unit,
      sym_bool,
      sym_f64,
      sym_symstring,
  [736] = 12,
    ACTIONS(67), 1,
      sym_lparen,
    ACTIONS(73), 1,
      sym_num,
    ACTIONS(77), 1,
      aux_sym_f64_token1,
    ACTIONS(79), 1,
      sym_ident,
    ACTIONS(81), 1,
      sym_string,
    ACTIONS(126), 1,
      sym_rparen,
    STATE(267), 1,
      sym_expr,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
    ACTIONS(71), 2,
      anon_sym_true,
      anon_sym_false,
    STATE(54), 2,
      sym_literal,
      sym_callexpr,
    ACTIONS(75), 3,
      anon_sym_NaN,
      anon_sym_inf,
      anon_sym_DASHinf,
    STATE(52), 4,
      sym_unit,
      sym_bool,
      sym_f64,
      sym_symstring,
  [781] = 11,
    ACTIONS(67), 1,
      sym_lparen,
    ACTIONS(73), 1,
      sym_num,
    ACTIONS(77), 1,
      aux_sym_f64_token1,
    ACTIONS(79), 1,
      sym_ident,
    ACTIONS(81), 1,
      sym_string,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
    ACTIONS(71), 2,
      anon_sym_true,
      anon_sym_false,
    STATE(15), 2,
      sym_expr,
      aux_sym_command_repeat2,
    STATE(54), 2,
      sym_literal,
      sym_callexpr,
    ACTIONS(75), 3,
      anon_sym_NaN,
      anon_sym_inf,
      anon_sym_DASHinf,
    STATE(52), 4,
      sym_unit,
      sym_bool,
      sym_f64,
      sym_symstring,
  [824] = 11,
    ACTIONS(67), 1,
      sym_lparen,
    ACTIONS(73), 1,
      sym_num,
    ACTIONS(77), 1,
      aux_sym_f64_token1,
    ACTIONS(79), 1,
      sym_ident,
    ACTIONS(81), 1,
      sym_string,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
    ACTIONS(71), 2,
      anon_sym_true,
      anon_sym_false,
    STATE(6), 2,
      sym_expr,
      aux_sym_command_repeat2,
    STATE(54), 2,
      sym_literal,
      sym_callexpr,
    ACTIONS(75), 3,
      anon_sym_NaN,
      anon_sym_inf,
      anon_sym_DASHinf,
    STATE(52), 4,
      sym_unit,
      sym_bool,
      sym_f64,
      sym_symstring,
  [867] = 11,
    ACTIONS(67), 1,
      sym_lparen,
    ACTIONS(73), 1,
      sym_num,
    ACTIONS(77), 1,
      aux_sym_f64_token1,
    ACTIONS(79), 1,
      sym_ident,
    ACTIONS(81), 1,
      sym_string,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
    ACTIONS(71), 2,
      anon_sym_true,
      anon_sym_false,
    STATE(10), 2,
      sym_expr,
      aux_sym_command_repeat2,
    STATE(54), 2,
      sym_literal,
      sym_callexpr,
    ACTIONS(75), 3,
      anon_sym_NaN,
      anon_sym_inf,
      anon_sym_DASHinf,
    STATE(52), 4,
      sym_unit,
      sym_bool,
      sym_f64,
      sym_symstring,
  [910] = 12,
    ACTIONS(67), 1,
      sym_lparen,
    ACTIONS(73), 1,
      sym_num,
    ACTIONS(77), 1,
      aux_sym_f64_token1,
    ACTIONS(79), 1,
      sym_ident,
    ACTIONS(81), 1,
      sym_string,
    ACTIONS(128), 1,
      anon_sym_COLONvariants,
    STATE(211), 1,
      sym_expr,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
    ACTIONS(71), 2,
      anon_sym_true,
      anon_sym_false,
    STATE(54), 2,
      sym_literal,
      sym_callexpr,
    ACTIONS(75), 3,
      anon_sym_NaN,
      anon_sym_inf,
      anon_sym_DASHinf,
    STATE(52), 4,
      sym_unit,
      sym_bool,
      sym_f64,
      sym_symstring,
  [955] = 11,
    ACTIONS(67), 1,
      sym_lparen,
    ACTIONS(73), 1,
      sym_num,
    ACTIONS(77), 1,
      aux_sym_f64_token1,
    ACTIONS(79), 1,
      sym_ident,
    ACTIONS(81), 1,
      sym_string,
    STATE(267), 1,
      sym_expr,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
    ACTIONS(71), 2,
      anon_sym_true,
      anon_sym_false,
    STATE(54), 2,
      sym_literal,
      sym_callexpr,
    ACTIONS(75), 3,
      anon_sym_NaN,
      anon_sym_inf,
      anon_sym_DASHinf,
    STATE(52), 4,
      sym_unit,
      sym_bool,
      sym_f64,
      sym_symstring,
  [997] = 11,
    ACTIONS(67), 1,
      sym_lparen,
    ACTIONS(73), 1,
      sym_num,
    ACTIONS(77), 1,
      aux_sym_f64_token1,
    ACTIONS(79), 1,
      sym_ident,
    ACTIONS(81), 1,
      sym_string,
    STATE(182), 1,
      sym_expr,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
    ACTIONS(71), 2,
      anon_sym_true,
      anon_sym_false,
    STATE(54), 2,
      sym_literal,
      sym_callexpr,
    ACTIONS(75), 3,
      anon_sym_NaN,
      anon_sym_inf,
      anon_sym_DASHinf,
    STATE(52), 4,
      sym_unit,
      sym_bool,
      sym_f64,
      sym_symstring,
  [1039] = 11,
    ACTIONS(67), 1,
      sym_lparen,
    ACTIONS(73), 1,
      sym_num,
    ACTIONS(77), 1,
      aux_sym_f64_token1,
    ACTIONS(79), 1,
      sym_ident,
    ACTIONS(81), 1,
      sym_string,
    STATE(210), 1,
      sym_expr,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
    ACTIONS(71), 2,
      anon_sym_true,
      anon_sym_false,
    STATE(54), 2,
      sym_literal,
      sym_callexpr,
    ACTIONS(75), 3,
      anon_sym_NaN,
      anon_sym_inf,
      anon_sym_DASHinf,
    STATE(52), 4,
      sym_unit,
      sym_bool,
      sym_f64,
      sym_symstring,
  [1081] = 11,
    ACTIONS(67), 1,
      sym_lparen,
    ACTIONS(73), 1,
      sym_num,
    ACTIONS(77), 1,
      aux_sym_f64_token1,
    ACTIONS(79), 1,
      sym_ident,
    ACTIONS(81), 1,
      sym_string,
    STATE(188), 1,
      sym_expr,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
    ACTIONS(71), 2,
      anon_sym_true,
      anon_sym_false,
    STATE(54), 2,
      sym_literal,
      sym_callexpr,
    ACTIONS(75), 3,
      anon_sym_NaN,
      anon_sym_inf,
      anon_sym_DASHinf,
    STATE(52), 4,
      sym_unit,
      sym_bool,
      sym_f64,
      sym_symstring,
  [1123] = 11,
    ACTIONS(67), 1,
      sym_lparen,
    ACTIONS(73), 1,
      sym_num,
    ACTIONS(77), 1,
      aux_sym_f64_token1,
    ACTIONS(79), 1,
      sym_ident,
    ACTIONS(81), 1,
      sym_string,
    STATE(228), 1,
      sym_expr,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
    ACTIONS(71), 2,
      anon_sym_true,
      anon_sym_false,
    STATE(54), 2,
      sym_literal,
      sym_callexpr,
    ACTIONS(75), 3,
      anon_sym_NaN,
      anon_sym_inf,
      anon_sym_DASHinf,
    STATE(52), 4,
      sym_unit,
      sym_bool,
      sym_f64,
      sym_symstring,
  [1165] = 11,
    ACTIONS(67), 1,
      sym_lparen,
    ACTIONS(73), 1,
      sym_num,
    ACTIONS(77), 1,
      aux_sym_f64_token1,
    ACTIONS(79), 1,
      sym_ident,
    ACTIONS(81), 1,
      sym_string,
    STATE(236), 1,
      sym_expr,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
    ACTIONS(71), 2,
      anon_sym_true,
      anon_sym_false,
    STATE(54), 2,
      sym_literal,
      sym_callexpr,
    ACTIONS(75), 3,
      anon_sym_NaN,
      anon_sym_inf,
      anon_sym_DASHinf,
    STATE(52), 4,
      sym_unit,
      sym_bool,
      sym_f64,
      sym_symstring,
  [1207] = 11,
    ACTIONS(67), 1,
      sym_lparen,
    ACTIONS(73), 1,
      sym_num,
    ACTIONS(77), 1,
      aux_sym_f64_token1,
    ACTIONS(79), 1,
      sym_ident,
    ACTIONS(81), 1,
      sym_string,
    STATE(41), 1,
      sym_expr,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
    ACTIONS(71), 2,
      anon_sym_true,
      anon_sym_false,
    STATE(54), 2,
      sym_literal,
      sym_callexpr,
    ACTIONS(75), 3,
      anon_sym_NaN,
      anon_sym_inf,
      anon_sym_DASHinf,
    STATE(52), 4,
      sym_unit,
      sym_bool,
      sym_f64,
      sym_symstring,
  [1249] = 11,
    ACTIONS(67), 1,
      sym_lparen,
    ACTIONS(73), 1,
      sym_num,
    ACTIONS(77), 1,
      aux_sym_f64_token1,
    ACTIONS(79), 1,
      sym_ident,
    ACTIONS(81), 1,
      sym_string,
    STATE(183), 1,
      sym_expr,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
    ACTIONS(71), 2,
      anon_sym_true,
      anon_sym_false,
    STATE(54), 2,
      sym_literal,
      sym_callexpr,
    ACTIONS(75), 3,
      anon_sym_NaN,
      anon_sym_inf,
      anon_sym_DASHinf,
    STATE(52), 4,
      sym_unit,
      sym_bool,
      sym_f64,
      sym_symstring,
  [1291] = 11,
    ACTIONS(67), 1,
      sym_lparen,
    ACTIONS(73), 1,
      sym_num,
    ACTIONS(77), 1,
      aux_sym_f64_token1,
    ACTIONS(79), 1,
      sym_ident,
    ACTIONS(81), 1,
      sym_string,
    STATE(216), 1,
      sym_expr,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
    ACTIONS(71), 2,
      anon_sym_true,
      anon_sym_false,
    STATE(54), 2,
      sym_literal,
      sym_callexpr,
    ACTIONS(75), 3,
      anon_sym_NaN,
      anon_sym_inf,
      anon_sym_DASHinf,
    STATE(52), 4,
      sym_unit,
      sym_bool,
      sym_f64,
      sym_symstring,
  [1333] = 11,
    ACTIONS(67), 1,
      sym_lparen,
    ACTIONS(73), 1,
      sym_num,
    ACTIONS(77), 1,
      aux_sym_f64_token1,
    ACTIONS(79), 1,
      sym_ident,
    ACTIONS(81), 1,
      sym_string,
    STATE(220), 1,
      sym_expr,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
    ACTIONS(71), 2,
      anon_sym_true,
      anon_sym_false,
    STATE(54), 2,
      sym_literal,
      sym_callexpr,
    ACTIONS(75), 3,
      anon_sym_NaN,
      anon_sym_inf,
      anon_sym_DASHinf,
    STATE(52), 4,
      sym_unit,
      sym_bool,
      sym_f64,
      sym_symstring,
  [1375] = 11,
    ACTIONS(67), 1,
      sym_lparen,
    ACTIONS(73), 1,
      sym_num,
    ACTIONS(77), 1,
      aux_sym_f64_token1,
    ACTIONS(79), 1,
      sym_ident,
    ACTIONS(81), 1,
      sym_string,
    STATE(246), 1,
      sym_expr,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
    ACTIONS(71), 2,
      anon_sym_true,
      anon_sym_false,
    STATE(54), 2,
      sym_literal,
      sym_callexpr,
    ACTIONS(75), 3,
      anon_sym_NaN,
      anon_sym_inf,
      anon_sym_DASHinf,
    STATE(52), 4,
      sym_unit,
      sym_bool,
      sym_f64,
      sym_symstring,
  [1417] = 11,
    ACTIONS(67), 1,
      sym_lparen,
    ACTIONS(73), 1,
      sym_num,
    ACTIONS(77), 1,
      aux_sym_f64_token1,
    ACTIONS(79), 1,
      sym_ident,
    ACTIONS(81), 1,
      sym_string,
    STATE(168), 1,
      sym_expr,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
    ACTIONS(71), 2,
      anon_sym_true,
      anon_sym_false,
    STATE(54), 2,
      sym_literal,
      sym_callexpr,
    ACTIONS(75), 3,
      anon_sym_NaN,
      anon_sym_inf,
      anon_sym_DASHinf,
    STATE(52), 4,
      sym_unit,
      sym_bool,
      sym_f64,
      sym_symstring,
  [1459] = 11,
    ACTIONS(67), 1,
      sym_lparen,
    ACTIONS(73), 1,
      sym_num,
    ACTIONS(77), 1,
      aux_sym_f64_token1,
    ACTIONS(79), 1,
      sym_ident,
    ACTIONS(81), 1,
      sym_string,
    STATE(251), 1,
      sym_expr,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
    ACTIONS(71), 2,
      anon_sym_true,
      anon_sym_false,
    STATE(54), 2,
      sym_literal,
      sym_callexpr,
    ACTIONS(75), 3,
      anon_sym_NaN,
      anon_sym_inf,
      anon_sym_DASHinf,
    STATE(52), 4,
      sym_unit,
      sym_bool,
      sym_f64,
      sym_symstring,
  [1501] = 11,
    ACTIONS(67), 1,
      sym_lparen,
    ACTIONS(73), 1,
      sym_num,
    ACTIONS(77), 1,
      aux_sym_f64_token1,
    ACTIONS(79), 1,
      sym_ident,
    ACTIONS(81), 1,
      sym_string,
    STATE(248), 1,
      sym_expr,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
    ACTIONS(71), 2,
      anon_sym_true,
      anon_sym_false,
    STATE(54), 2,
      sym_literal,
      sym_callexpr,
    ACTIONS(75), 3,
      anon_sym_NaN,
      anon_sym_inf,
      anon_sym_DASHinf,
    STATE(52), 4,
      sym_unit,
      sym_bool,
      sym_f64,
      sym_symstring,
  [1543] = 11,
    ACTIONS(67), 1,
      sym_lparen,
    ACTIONS(73), 1,
      sym_num,
    ACTIONS(77), 1,
      aux_sym_f64_token1,
    ACTIONS(79), 1,
      sym_ident,
    ACTIONS(81), 1,
      sym_string,
    STATE(173), 1,
      sym_expr,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
    ACTIONS(71), 2,
      anon_sym_true,
      anon_sym_false,
    STATE(54), 2,
      sym_literal,
      sym_callexpr,
    ACTIONS(75), 3,
      anon_sym_NaN,
      anon_sym_inf,
      anon_sym_DASHinf,
    STATE(52), 4,
      sym_unit,
      sym_bool,
      sym_f64,
      sym_symstring,
  [1585] = 11,
    ACTIONS(67), 1,
      sym_lparen,
    ACTIONS(73), 1,
      sym_num,
    ACTIONS(77), 1,
      aux_sym_f64_token1,
    ACTIONS(79), 1,
      sym_ident,
    ACTIONS(81), 1,
      sym_string,
    STATE(243), 1,
      sym_expr,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
    ACTIONS(71), 2,
      anon_sym_true,
      anon_sym_false,
    STATE(54), 2,
      sym_literal,
      sym_callexpr,
    ACTIONS(75), 3,
      anon_sym_NaN,
      anon_sym_inf,
      anon_sym_DASHinf,
    STATE(52), 4,
      sym_unit,
      sym_bool,
      sym_f64,
      sym_symstring,
  [1627] = 11,
    ACTIONS(67), 1,
      sym_lparen,
    ACTIONS(73), 1,
      sym_num,
    ACTIONS(77), 1,
      aux_sym_f64_token1,
    ACTIONS(79), 1,
      sym_ident,
    ACTIONS(81), 1,
      sym_string,
    STATE(22), 1,
      sym_expr,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
    ACTIONS(71), 2,
      anon_sym_true,
      anon_sym_false,
    STATE(54), 2,
      sym_literal,
      sym_callexpr,
    ACTIONS(75), 3,
      anon_sym_NaN,
      anon_sym_inf,
      anon_sym_DASHinf,
    STATE(52), 4,
      sym_unit,
      sym_bool,
      sym_f64,
      sym_symstring,
  [1669] = 11,
    ACTIONS(67), 1,
      sym_lparen,
    ACTIONS(73), 1,
      sym_num,
    ACTIONS(77), 1,
      aux_sym_f64_token1,
    ACTIONS(79), 1,
      sym_ident,
    ACTIONS(81), 1,
      sym_string,
    STATE(240), 1,
      sym_expr,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
    ACTIONS(71), 2,
      anon_sym_true,
      anon_sym_false,
    STATE(54), 2,
      sym_literal,
      sym_callexpr,
    ACTIONS(75), 3,
      anon_sym_NaN,
      anon_sym_inf,
      anon_sym_DASHinf,
    STATE(52), 4,
      sym_unit,
      sym_bool,
      sym_f64,
      sym_symstring,
  [1711] = 11,
    ACTIONS(67), 1,
      sym_lparen,
    ACTIONS(73), 1,
      sym_num,
    ACTIONS(77), 1,
      aux_sym_f64_token1,
    ACTIONS(79), 1,
      sym_ident,
    ACTIONS(81), 1,
      sym_string,
    STATE(17), 1,
      sym_expr,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
    ACTIONS(71), 2,
      anon_sym_true,
      anon_sym_false,
    STATE(54), 2,
      sym_literal,
      sym_callexpr,
    ACTIONS(75), 3,
      anon_sym_NaN,
      anon_sym_inf,
      anon_sym_DASHinf,
    STATE(52), 4,
      sym_unit,
      sym_bool,
      sym_f64,
      sym_symstring,
  [1753] = 11,
    ACTIONS(67), 1,
      sym_lparen,
    ACTIONS(73), 1,
      sym_num,
    ACTIONS(77), 1,
      aux_sym_f64_token1,
    ACTIONS(79), 1,
      sym_ident,
    ACTIONS(81), 1,
      sym_string,
    STATE(161), 1,
      sym_expr,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
    ACTIONS(71), 2,
      anon_sym_true,
      anon_sym_false,
    STATE(54), 2,
      sym_literal,
      sym_callexpr,
    ACTIONS(75), 3,
      anon_sym_NaN,
      anon_sym_inf,
      anon_sym_DASHinf,
    STATE(52), 4,
      sym_unit,
      sym_bool,
      sym_f64,
      sym_symstring,
  [1795] = 11,
    ACTIONS(67), 1,
      sym_lparen,
    ACTIONS(73), 1,
      sym_num,
    ACTIONS(77), 1,
      aux_sym_f64_token1,
    ACTIONS(79), 1,
      sym_ident,
    ACTIONS(81), 1,
      sym_string,
    STATE(175), 1,
      sym_expr,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
    ACTIONS(71), 2,
      anon_sym_true,
      anon_sym_false,
    STATE(54), 2,
      sym_literal,
      sym_callexpr,
    ACTIONS(75), 3,
      anon_sym_NaN,
      anon_sym_inf,
      anon_sym_DASHinf,
    STATE(52), 4,
      sym_unit,
      sym_bool,
      sym_f64,
      sym_symstring,
  [1837] = 11,
    ACTIONS(67), 1,
      sym_lparen,
    ACTIONS(73), 1,
      sym_num,
    ACTIONS(77), 1,
      aux_sym_f64_token1,
    ACTIONS(79), 1,
      sym_ident,
    ACTIONS(81), 1,
      sym_string,
    STATE(225), 1,
      sym_expr,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
    ACTIONS(71), 2,
      anon_sym_true,
      anon_sym_false,
    STATE(54), 2,
      sym_literal,
      sym_callexpr,
    ACTIONS(75), 3,
      anon_sym_NaN,
      anon_sym_inf,
      anon_sym_DASHinf,
    STATE(52), 4,
      sym_unit,
      sym_bool,
      sym_f64,
      sym_symstring,
  [1879] = 11,
    ACTIONS(67), 1,
      sym_lparen,
    ACTIONS(73), 1,
      sym_num,
    ACTIONS(77), 1,
      aux_sym_f64_token1,
    ACTIONS(79), 1,
      sym_ident,
    ACTIONS(81), 1,
      sym_string,
    STATE(252), 1,
      sym_expr,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
    ACTIONS(71), 2,
      anon_sym_true,
      anon_sym_false,
    STATE(54), 2,
      sym_literal,
      sym_callexpr,
    ACTIONS(75), 3,
      anon_sym_NaN,
      anon_sym_inf,
      anon_sym_DASHinf,
    STATE(52), 4,
      sym_unit,
      sym_bool,
      sym_f64,
      sym_symstring,
  [1921] = 11,
    ACTIONS(67), 1,
      sym_lparen,
    ACTIONS(73), 1,
      sym_num,
    ACTIONS(77), 1,
      aux_sym_f64_token1,
    ACTIONS(79), 1,
      sym_ident,
    ACTIONS(81), 1,
      sym_string,
    STATE(229), 1,
      sym_expr,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
    ACTIONS(71), 2,
      anon_sym_true,
      anon_sym_false,
    STATE(54), 2,
      sym_literal,
      sym_callexpr,
    ACTIONS(75), 3,
      anon_sym_NaN,
      anon_sym_inf,
      anon_sym_DASHinf,
    STATE(52), 4,
      sym_unit,
      sym_bool,
      sym_f64,
      sym_symstring,
  [1963] = 11,
    ACTIONS(67), 1,
      sym_lparen,
    ACTIONS(73), 1,
      sym_num,
    ACTIONS(77), 1,
      aux_sym_f64_token1,
    ACTIONS(79), 1,
      sym_ident,
    ACTIONS(81), 1,
      sym_string,
    STATE(179), 1,
      sym_expr,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
    ACTIONS(71), 2,
      anon_sym_true,
      anon_sym_false,
    STATE(54), 2,
      sym_literal,
      sym_callexpr,
    ACTIONS(75), 3,
      anon_sym_NaN,
      anon_sym_inf,
      anon_sym_DASHinf,
    STATE(52), 4,
      sym_unit,
      sym_bool,
      sym_f64,
      sym_symstring,
  [2005] = 11,
    ACTIONS(67), 1,
      sym_lparen,
    ACTIONS(73), 1,
      sym_num,
    ACTIONS(77), 1,
      aux_sym_f64_token1,
    ACTIONS(79), 1,
      sym_ident,
    ACTIONS(81), 1,
      sym_string,
    STATE(223), 1,
      sym_expr,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
    ACTIONS(71), 2,
      anon_sym_true,
      anon_sym_false,
    STATE(54), 2,
      sym_literal,
      sym_callexpr,
    ACTIONS(75), 3,
      anon_sym_NaN,
      anon_sym_inf,
      anon_sym_DASHinf,
    STATE(52), 4,
      sym_unit,
      sym_bool,
      sym_f64,
      sym_symstring,
  [2047] = 3,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
    ACTIONS(132), 7,
      anon_sym_true,
      anon_sym_false,
      sym_num,
      anon_sym_NaN,
      anon_sym_inf,
      anon_sym_DASHinf,
      sym_ident,
    ACTIONS(130), 8,
      ts_builtin_sym_end,
      sym_lparen,
      sym_rparen,
      anon_sym_COLONdefault,
      anon_sym_COLONruleset,
      anon_sym_COLONwhen,
      aux_sym_f64_token1,
      sym_string,
  [2071] = 3,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
    ACTIONS(136), 7,
      anon_sym_true,
      anon_sym_false,
      sym_num,
      anon_sym_NaN,
      anon_sym_inf,
      anon_sym_DASHinf,
      sym_ident,
    ACTIONS(134), 8,
      ts_builtin_sym_end,
      sym_lparen,
      sym_rparen,
      anon_sym_COLONdefault,
      anon_sym_COLONruleset,
      anon_sym_COLONwhen,
      aux_sym_f64_token1,
      sym_string,
  [2095] = 3,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
    ACTIONS(138), 7,
      sym_lparen,
      sym_rparen,
      anon_sym_COLONdefault,
      anon_sym_COLONruleset,
      anon_sym_COLONwhen,
      aux_sym_f64_token1,
      sym_string,
    ACTIONS(140), 7,
      anon_sym_true,
      anon_sym_false,
      sym_num,
      anon_sym_NaN,
      anon_sym_inf,
      anon_sym_DASHinf,
      sym_ident,
  [2118] = 3,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
    ACTIONS(142), 7,
      sym_lparen,
      sym_rparen,
      anon_sym_COLONdefault,
      anon_sym_COLONruleset,
      anon_sym_COLONwhen,
      aux_sym_f64_token1,
      sym_string,
    ACTIONS(144), 7,
      anon_sym_true,
      anon_sym_false,
      sym_num,
      anon_sym_NaN,
      anon_sym_inf,
      anon_sym_DASHinf,
      sym_ident,
  [2141] = 3,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
    ACTIONS(146), 7,
      sym_lparen,
      sym_rparen,
      anon_sym_COLONdefault,
      anon_sym_COLONruleset,
      anon_sym_COLONwhen,
      aux_sym_f64_token1,
      sym_string,
    ACTIONS(148), 7,
      anon_sym_true,
      anon_sym_false,
      sym_num,
      anon_sym_NaN,
      anon_sym_inf,
      anon_sym_DASHinf,
      sym_ident,
  [2164] = 3,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
    ACTIONS(150), 7,
      sym_lparen,
      sym_rparen,
      anon_sym_COLONdefault,
      anon_sym_COLONruleset,
      anon_sym_COLONwhen,
      aux_sym_f64_token1,
      sym_string,
    ACTIONS(152), 7,
      anon_sym_true,
      anon_sym_false,
      sym_num,
      anon_sym_NaN,
      anon_sym_inf,
      anon_sym_DASHinf,
      sym_ident,
  [2187] = 3,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
    ACTIONS(154), 7,
      sym_lparen,
      sym_rparen,
      anon_sym_COLONdefault,
      anon_sym_COLONruleset,
      anon_sym_COLONwhen,
      aux_sym_f64_token1,
      sym_string,
    ACTIONS(156), 7,
      anon_sym_true,
      anon_sym_false,
      sym_num,
      anon_sym_NaN,
      anon_sym_inf,
      anon_sym_DASHinf,
      sym_ident,
  [2210] = 3,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
    ACTIONS(158), 7,
      sym_lparen,
      sym_rparen,
      anon_sym_COLONdefault,
      anon_sym_COLONruleset,
      anon_sym_COLONwhen,
      aux_sym_f64_token1,
      sym_string,
    ACTIONS(160), 7,
      anon_sym_true,
      anon_sym_false,
      sym_num,
      anon_sym_NaN,
      anon_sym_inf,
      anon_sym_DASHinf,
      sym_ident,
  [2233] = 4,
    ACTIONS(164), 1,
      sym_rparen,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
    ACTIONS(162), 3,
      sym_lparen,
      aux_sym_f64_token1,
      sym_string,
    ACTIONS(166), 7,
      anon_sym_true,
      anon_sym_false,
      sym_num,
      anon_sym_NaN,
      anon_sym_inf,
      anon_sym_DASHinf,
      sym_ident,
  [2255] = 3,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
    ACTIONS(168), 3,
      sym_lparen,
      aux_sym_f64_token1,
      sym_string,
    ACTIONS(170), 7,
      anon_sym_true,
      anon_sym_false,
      sym_num,
      anon_sym_NaN,
      anon_sym_inf,
      anon_sym_DASHinf,
      sym_ident,
  [2274] = 3,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
    ACTIONS(172), 3,
      sym_lparen,
      aux_sym_f64_token1,
      sym_string,
    ACTIONS(174), 7,
      anon_sym_true,
      anon_sym_false,
      sym_num,
      anon_sym_NaN,
      anon_sym_inf,
      anon_sym_DASHinf,
      sym_ident,
  [2293] = 3,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
    ACTIONS(176), 3,
      sym_lparen,
      aux_sym_f64_token1,
      sym_string,
    ACTIONS(178), 7,
      anon_sym_true,
      anon_sym_false,
      sym_num,
      anon_sym_NaN,
      anon_sym_inf,
      anon_sym_DASHinf,
      sym_ident,
  [2312] = 3,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
    ACTIONS(180), 3,
      sym_lparen,
      aux_sym_f64_token1,
      sym_string,
    ACTIONS(182), 7,
      anon_sym_true,
      anon_sym_false,
      sym_num,
      anon_sym_NaN,
      anon_sym_inf,
      anon_sym_DASHinf,
      sym_ident,
  [2331] = 3,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
    ACTIONS(184), 3,
      sym_lparen,
      aux_sym_f64_token1,
      sym_string,
    ACTIONS(186), 7,
      anon_sym_true,
      anon_sym_false,
      sym_num,
      anon_sym_NaN,
      anon_sym_inf,
      anon_sym_DASHinf,
      sym_ident,
  [2350] = 8,
    ACTIONS(9), 1,
      sym_lparen,
    ACTIONS(57), 1,
      anon_sym_delete,
    ACTIONS(59), 1,
      anon_sym_union,
    ACTIONS(61), 1,
      anon_sym_panic,
    ACTIONS(63), 1,
      anon_sym_extract,
    ACTIONS(65), 1,
      sym_ident,
    ACTIONS(188), 1,
      anon_sym_let,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
  [2376] = 8,
    ACTIONS(116), 1,
      sym_rparen,
    ACTIONS(190), 1,
      anon_sym_COLONunextractable,
    ACTIONS(192), 1,
      anon_sym_COLONon_merge,
    ACTIONS(194), 1,
      anon_sym_COLONmerge,
    ACTIONS(196), 1,
      anon_sym_COLONdefault,
    ACTIONS(198), 1,
      anon_sym_COLONcost,
    STATE(100), 1,
      sym_cost,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
  [2402] = 6,
    ACTIONS(7), 1,
      sym_lparen,
    ACTIONS(200), 1,
      ts_builtin_sym_end,
    STATE(131), 1,
      sym_callexpr,
    STATE(155), 1,
      sym_nonletaction,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
    STATE(74), 2,
      sym_command,
      aux_sym_source_file_repeat1,
  [2423] = 6,
    ACTIONS(202), 1,
      sym_lparen,
    ACTIONS(204), 1,
      sym_rparen,
    STATE(131), 1,
      sym_callexpr,
    STATE(170), 1,
      sym_nonletaction,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
    STATE(66), 2,
      sym_action,
      aux_sym_command_repeat3,
  [2444] = 6,
    ACTIONS(206), 1,
      sym_lparen,
    ACTIONS(209), 1,
      sym_rparen,
    STATE(131), 1,
      sym_callexpr,
    STATE(170), 1,
      sym_nonletaction,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
    STATE(66), 2,
      sym_action,
      aux_sym_command_repeat3,
  [2465] = 2,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
    ACTIONS(211), 6,
      sym_rparen,
      anon_sym_COLONunextractable,
      anon_sym_COLONon_merge,
      anon_sym_COLONmerge,
      anon_sym_COLONdefault,
      anon_sym_COLONcost,
  [2478] = 6,
    ACTIONS(202), 1,
      sym_lparen,
    ACTIONS(213), 1,
      sym_rparen,
    STATE(131), 1,
      sym_callexpr,
    STATE(170), 1,
      sym_nonletaction,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
    STATE(70), 2,
      sym_action,
      aux_sym_command_repeat3,
  [2499] = 6,
    ACTIONS(202), 1,
      sym_lparen,
    ACTIONS(215), 1,
      sym_rparen,
    STATE(131), 1,
      sym_callexpr,
    STATE(170), 1,
      sym_nonletaction,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
    STATE(66), 2,
      sym_action,
      aux_sym_command_repeat3,
  [2520] = 6,
    ACTIONS(202), 1,
      sym_lparen,
    ACTIONS(217), 1,
      sym_rparen,
    STATE(131), 1,
      sym_callexpr,
    STATE(170), 1,
      sym_nonletaction,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
    STATE(66), 2,
      sym_action,
      aux_sym_command_repeat3,
  [2541] = 6,
    ACTIONS(202), 1,
      sym_lparen,
    ACTIONS(217), 1,
      sym_rparen,
    STATE(131), 1,
      sym_callexpr,
    STATE(170), 1,
      sym_nonletaction,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
    STATE(69), 2,
      sym_action,
      aux_sym_command_repeat3,
  [2562] = 2,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
    ACTIONS(219), 6,
      sym_rparen,
      anon_sym_COLONunextractable,
      anon_sym_COLONon_merge,
      anon_sym_COLONmerge,
      anon_sym_COLONdefault,
      anon_sym_COLONcost,
  [2575] = 6,
    ACTIONS(202), 1,
      sym_lparen,
    ACTIONS(215), 1,
      sym_rparen,
    STATE(131), 1,
      sym_callexpr,
    STATE(170), 1,
      sym_nonletaction,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
    STATE(77), 2,
      sym_action,
      aux_sym_command_repeat3,
  [2596] = 6,
    ACTIONS(221), 1,
      ts_builtin_sym_end,
    ACTIONS(223), 1,
      sym_lparen,
    STATE(131), 1,
      sym_callexpr,
    STATE(155), 1,
      sym_nonletaction,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
    STATE(74), 2,
      sym_command,
      aux_sym_source_file_repeat1,
  [2617] = 6,
    ACTIONS(202), 1,
      sym_lparen,
    ACTIONS(226), 1,
      sym_rparen,
    STATE(131), 1,
      sym_callexpr,
    STATE(170), 1,
      sym_nonletaction,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
    STATE(78), 2,
      sym_action,
      aux_sym_command_repeat3,
  [2638] = 6,
    ACTIONS(202), 1,
      sym_lparen,
    ACTIONS(228), 1,
      sym_rparen,
    STATE(131), 1,
      sym_callexpr,
    STATE(170), 1,
      sym_nonletaction,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
    STATE(65), 2,
      sym_action,
      aux_sym_command_repeat3,
  [2659] = 6,
    ACTIONS(202), 1,
      sym_lparen,
    ACTIONS(230), 1,
      sym_rparen,
    STATE(131), 1,
      sym_callexpr,
    STATE(170), 1,
      sym_nonletaction,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
    STATE(66), 2,
      sym_action,
      aux_sym_command_repeat3,
  [2680] = 6,
    ACTIONS(202), 1,
      sym_lparen,
    ACTIONS(228), 1,
      sym_rparen,
    STATE(131), 1,
      sym_callexpr,
    STATE(170), 1,
      sym_nonletaction,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
    STATE(66), 2,
      sym_action,
      aux_sym_command_repeat3,
  [2701] = 5,
    ACTIONS(232), 1,
      sym_lparen,
    ACTIONS(234), 1,
      sym_rparen,
    STATE(191), 1,
      sym_callexpr,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
    STATE(96), 2,
      sym_fact,
      aux_sym_command_repeat5,
  [2719] = 5,
    ACTIONS(236), 1,
      sym_lparen,
    ACTIONS(238), 1,
      sym_rparen,
    ACTIONS(240), 1,
      sym_ident,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
    STATE(97), 2,
      sym_schedule,
      aux_sym_command_repeat7,
  [2737] = 5,
    ACTIONS(232), 1,
      sym_lparen,
    ACTIONS(242), 1,
      sym_rparen,
    STATE(191), 1,
      sym_callexpr,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
    STATE(92), 2,
      sym_fact,
      aux_sym_command_repeat5,
  [2755] = 5,
    ACTIONS(232), 1,
      sym_lparen,
    ACTIONS(238), 1,
      sym_rparen,
    STATE(191), 1,
      sym_callexpr,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
    STATE(94), 2,
      sym_fact,
      aux_sym_command_repeat5,
  [2773] = 5,
    ACTIONS(232), 1,
      sym_lparen,
    ACTIONS(244), 1,
      sym_rparen,
    STATE(191), 1,
      sym_callexpr,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
    STATE(96), 2,
      sym_fact,
      aux_sym_command_repeat5,
  [2791] = 5,
    ACTIONS(232), 1,
      sym_lparen,
    ACTIONS(238), 1,
      sym_rparen,
    STATE(191), 1,
      sym_callexpr,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
    STATE(96), 2,
      sym_fact,
      aux_sym_command_repeat5,
  [2809] = 5,
    ACTIONS(234), 1,
      sym_rparen,
    ACTIONS(236), 1,
      sym_lparen,
    ACTIONS(240), 1,
      sym_ident,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
    STATE(97), 2,
      sym_schedule,
      aux_sym_command_repeat7,
  [2827] = 5,
    ACTIONS(116), 1,
      sym_rparen,
    ACTIONS(232), 1,
      sym_lparen,
    STATE(191), 1,
      sym_callexpr,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
    STATE(110), 2,
      sym_fact,
      aux_sym_command_repeat5,
  [2845] = 5,
    ACTIONS(122), 1,
      sym_rparen,
    ACTIONS(232), 1,
      sym_lparen,
    STATE(191), 1,
      sym_callexpr,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
    STATE(96), 2,
      sym_fact,
      aux_sym_command_repeat5,
  [2863] = 2,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
    ACTIONS(246), 5,
      sym_rparen,
      anon_sym_COLONunextractable,
      anon_sym_COLONon_merge,
      anon_sym_COLONmerge,
      anon_sym_COLONdefault,
  [2875] = 5,
    ACTIONS(236), 1,
      sym_lparen,
    ACTIONS(240), 1,
      sym_ident,
    ACTIONS(248), 1,
      sym_rparen,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
    STATE(101), 2,
      sym_schedule,
      aux_sym_command_repeat7,
  [2893] = 5,
    ACTIONS(232), 1,
      sym_lparen,
    ACTIONS(250), 1,
      sym_rparen,
    STATE(191), 1,
      sym_callexpr,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
    STATE(96), 2,
      sym_fact,
      aux_sym_command_repeat5,
  [2911] = 5,
    ACTIONS(232), 1,
      sym_lparen,
    ACTIONS(252), 1,
      sym_rparen,
    STATE(191), 1,
      sym_callexpr,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
    STATE(84), 2,
      sym_fact,
      aux_sym_command_repeat5,
  [2929] = 5,
    ACTIONS(232), 1,
      sym_lparen,
    ACTIONS(254), 1,
      sym_rparen,
    STATE(191), 1,
      sym_callexpr,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
    STATE(96), 2,
      sym_fact,
      aux_sym_command_repeat5,
  [2947] = 5,
    ACTIONS(87), 1,
      sym_rparen,
    ACTIONS(232), 1,
      sym_lparen,
    STATE(191), 1,
      sym_callexpr,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
    STATE(87), 2,
      sym_fact,
      aux_sym_command_repeat5,
  [2965] = 5,
    ACTIONS(232), 1,
      sym_lparen,
    ACTIONS(256), 1,
      sym_rparen,
    STATE(191), 1,
      sym_callexpr,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
    STATE(96), 2,
      sym_fact,
      aux_sym_command_repeat5,
  [2983] = 5,
    ACTIONS(232), 1,
      sym_lparen,
    ACTIONS(258), 1,
      sym_rparen,
    STATE(191), 1,
      sym_callexpr,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
    STATE(83), 2,
      sym_fact,
      aux_sym_command_repeat5,
  [3001] = 5,
    ACTIONS(260), 1,
      sym_lparen,
    ACTIONS(263), 1,
      sym_rparen,
    STATE(191), 1,
      sym_callexpr,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
    STATE(96), 2,
      sym_fact,
      aux_sym_command_repeat5,
  [3019] = 5,
    ACTIONS(265), 1,
      sym_lparen,
    ACTIONS(268), 1,
      sym_rparen,
    ACTIONS(270), 1,
      sym_ident,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
    STATE(97), 2,
      sym_schedule,
      aux_sym_command_repeat7,
  [3037] = 5,
    ACTIONS(236), 1,
      sym_lparen,
    ACTIONS(240), 1,
      sym_ident,
    ACTIONS(273), 1,
      sym_rparen,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
    STATE(104), 2,
      sym_schedule,
      aux_sym_command_repeat7,
  [3055] = 6,
    ACTIONS(198), 1,
      anon_sym_COLONcost,
    ACTIONS(275), 1,
      sym_rparen,
    ACTIONS(277), 1,
      sym_type,
    STATE(119), 1,
      aux_sym_command_repeat4,
    STATE(234), 1,
      sym_cost,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
  [3075] = 6,
    ACTIONS(87), 1,
      sym_rparen,
    ACTIONS(279), 1,
      anon_sym_COLONunextractable,
    ACTIONS(281), 1,
      anon_sym_COLONon_merge,
    ACTIONS(283), 1,
      anon_sym_COLONmerge,
    ACTIONS(285), 1,
      anon_sym_COLONdefault,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
  [3095] = 5,
    ACTIONS(236), 1,
      sym_lparen,
    ACTIONS(240), 1,
      sym_ident,
    ACTIONS(252), 1,
      sym_rparen,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
    STATE(97), 2,
      sym_schedule,
      aux_sym_command_repeat7,
  [3113] = 5,
    ACTIONS(232), 1,
      sym_lparen,
    ACTIONS(287), 1,
      sym_rparen,
    STATE(191), 1,
      sym_callexpr,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
    STATE(108), 2,
      sym_fact,
      aux_sym_command_repeat5,
  [3131] = 5,
    ACTIONS(236), 1,
      sym_lparen,
    ACTIONS(240), 1,
      sym_ident,
    ACTIONS(252), 1,
      sym_rparen,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
    STATE(80), 2,
      sym_schedule,
      aux_sym_command_repeat7,
  [3149] = 5,
    ACTIONS(236), 1,
      sym_lparen,
    ACTIONS(240), 1,
      sym_ident,
    ACTIONS(287), 1,
      sym_rparen,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
    STATE(97), 2,
      sym_schedule,
      aux_sym_command_repeat7,
  [3167] = 5,
    ACTIONS(236), 1,
      sym_lparen,
    ACTIONS(240), 1,
      sym_ident,
    ACTIONS(287), 1,
      sym_rparen,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
    STATE(112), 2,
      sym_schedule,
      aux_sym_command_repeat7,
  [3185] = 5,
    ACTIONS(236), 1,
      sym_lparen,
    ACTIONS(240), 1,
      sym_ident,
    ACTIONS(289), 1,
      sym_rparen,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
    STATE(85), 2,
      sym_schedule,
      aux_sym_command_repeat7,
  [3203] = 6,
    ACTIONS(198), 1,
      anon_sym_COLONcost,
    ACTIONS(291), 1,
      sym_rparen,
    ACTIONS(293), 1,
      sym_type,
    STATE(99), 1,
      aux_sym_command_repeat4,
    STATE(208), 1,
      sym_cost,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
  [3223] = 5,
    ACTIONS(232), 1,
      sym_lparen,
    ACTIONS(295), 1,
      sym_rparen,
    STATE(191), 1,
      sym_callexpr,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
    STATE(96), 2,
      sym_fact,
      aux_sym_command_repeat5,
  [3241] = 5,
    ACTIONS(232), 1,
      sym_lparen,
    ACTIONS(289), 1,
      sym_rparen,
    STATE(191), 1,
      sym_callexpr,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
    STATE(79), 2,
      sym_fact,
      aux_sym_command_repeat5,
  [3259] = 5,
    ACTIONS(87), 1,
      sym_rparen,
    ACTIONS(232), 1,
      sym_lparen,
    STATE(191), 1,
      sym_callexpr,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
    STATE(96), 2,
      sym_fact,
      aux_sym_command_repeat5,
  [3277] = 5,
    ACTIONS(232), 1,
      sym_lparen,
    ACTIONS(295), 1,
      sym_rparen,
    STATE(191), 1,
      sym_callexpr,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
    STATE(90), 2,
      sym_fact,
      aux_sym_command_repeat5,
  [3295] = 5,
    ACTIONS(236), 1,
      sym_lparen,
    ACTIONS(240), 1,
      sym_ident,
    ACTIONS(295), 1,
      sym_rparen,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
    STATE(97), 2,
      sym_schedule,
      aux_sym_command_repeat7,
  [3313] = 4,
    ACTIONS(116), 1,
      sym_rparen,
    ACTIONS(297), 1,
      sym_lparen,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
    STATE(122), 2,
      sym_variant,
      aux_sym_command_repeat1,
  [3328] = 5,
    ACTIONS(87), 1,
      sym_rparen,
    ACTIONS(281), 1,
      anon_sym_COLONon_merge,
    ACTIONS(283), 1,
      anon_sym_COLONmerge,
    ACTIONS(285), 1,
      anon_sym_COLONdefault,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
  [3345] = 4,
    ACTIONS(299), 1,
      anon_sym_run,
    ACTIONS(303), 1,
      anon_sym_repeat,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
    ACTIONS(301), 2,
      anon_sym_saturate,
      anon_sym_seq,
  [3360] = 5,
    ACTIONS(122), 1,
      sym_rparen,
    ACTIONS(305), 1,
      anon_sym_COLONon_merge,
    ACTIONS(307), 1,
      anon_sym_COLONmerge,
    ACTIONS(309), 1,
      anon_sym_COLONdefault,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
  [3377] = 4,
    ACTIONS(311), 1,
      sym_lparen,
    ACTIONS(313), 1,
      sym_rparen,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
    STATE(123), 2,
      sym_identsort,
      aux_sym_command_repeat6,
  [3392] = 4,
    ACTIONS(311), 1,
      sym_lparen,
    ACTIONS(315), 1,
      sym_rparen,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
    STATE(117), 2,
      sym_identsort,
      aux_sym_command_repeat6,
  [3407] = 4,
    ACTIONS(319), 1,
      sym_type,
    STATE(119), 1,
      aux_sym_command_repeat4,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
    ACTIONS(317), 2,
      sym_rparen,
      anon_sym_COLONcost,
  [3422] = 5,
    ACTIONS(7), 1,
      sym_lparen,
    STATE(131), 1,
      sym_callexpr,
    STATE(155), 1,
      sym_nonletaction,
    STATE(211), 1,
      sym_command,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
  [3439] = 4,
    ACTIONS(322), 1,
      anon_sym_run,
    ACTIONS(326), 1,
      anon_sym_repeat,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
    ACTIONS(324), 2,
      anon_sym_saturate,
      anon_sym_seq,
  [3454] = 4,
    ACTIONS(328), 1,
      sym_lparen,
    ACTIONS(331), 1,
      sym_rparen,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
    STATE(122), 2,
      sym_variant,
      aux_sym_command_repeat1,
  [3469] = 4,
    ACTIONS(333), 1,
      sym_lparen,
    ACTIONS(336), 1,
      sym_rparen,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
    STATE(123), 2,
      sym_identsort,
      aux_sym_command_repeat6,
  [3484] = 4,
    ACTIONS(234), 1,
      sym_rparen,
    ACTIONS(297), 1,
      sym_lparen,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
    STATE(113), 2,
      sym_variant,
      aux_sym_command_repeat1,
  [3499] = 4,
    ACTIONS(277), 1,
      sym_type,
    ACTIONS(338), 1,
      sym_rparen,
    STATE(119), 1,
      aux_sym_command_repeat4,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
  [3513] = 4,
    ACTIONS(340), 1,
      sym_rparen,
    ACTIONS(342), 1,
      anon_sym_COLONmerge,
    ACTIONS(344), 1,
      anon_sym_COLONdefault,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
  [3527] = 2,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
    ACTIONS(184), 3,
      sym_lparen,
      sym_rparen,
      sym_ident,
  [3537] = 2,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
    ACTIONS(346), 3,
      ts_builtin_sym_end,
      sym_lparen,
      sym_rparen,
  [3547] = 2,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
    ACTIONS(348), 3,
      ts_builtin_sym_end,
      sym_lparen,
      sym_rparen,
  [3557] = 4,
    ACTIONS(89), 1,
      sym_rparen,
    ACTIONS(277), 1,
      sym_type,
    STATE(119), 1,
      aux_sym_command_repeat4,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
  [3571] = 2,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
    ACTIONS(350), 3,
      ts_builtin_sym_end,
      sym_lparen,
      sym_rparen,
  [3581] = 2,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
    ACTIONS(352), 3,
      ts_builtin_sym_end,
      sym_lparen,
      sym_rparen,
  [3591] = 2,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
    ACTIONS(354), 3,
      ts_builtin_sym_end,
      sym_lparen,
      sym_rparen,
  [3601] = 2,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
    ACTIONS(356), 3,
      ts_builtin_sym_end,
      sym_lparen,
      sym_rparen,
  [3611] = 4,
    ACTIONS(358), 1,
      sym_rparen,
    ACTIONS(360), 1,
      anon_sym_COLONmerge,
    ACTIONS(362), 1,
      anon_sym_COLONdefault,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
  [3625] = 4,
    ACTIONS(364), 1,
      sym_lparen,
    ACTIONS(366), 1,
      sym_ident,
    STATE(43), 1,
      sym_schedule,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
  [3639] = 2,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
    ACTIONS(368), 3,
      ts_builtin_sym_end,
      sym_lparen,
      sym_rparen,
  [3649] = 2,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
    ACTIONS(180), 3,
      sym_lparen,
      sym_rparen,
      sym_ident,
  [3659] = 4,
    ACTIONS(370), 1,
      sym_rparen,
    ACTIONS(372), 1,
      anon_sym_COLONmerge,
    ACTIONS(374), 1,
      anon_sym_COLONdefault,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
  [3673] = 2,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
    ACTIONS(376), 3,
      ts_builtin_sym_end,
      sym_lparen,
      sym_rparen,
  [3683] = 2,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
    ACTIONS(378), 3,
      ts_builtin_sym_end,
      sym_lparen,
      sym_rparen,
  [3693] = 4,
    ACTIONS(340), 1,
      sym_rparen,
    ACTIONS(380), 1,
      anon_sym_COLONruleset,
    ACTIONS(382), 1,
      anon_sym_COLONname,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
  [3707] = 2,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
    ACTIONS(176), 3,
      sym_lparen,
      sym_rparen,
      sym_ident,
  [3717] = 4,
    ACTIONS(384), 1,
      sym_rparen,
    ACTIONS(386), 1,
      anon_sym_COLONruleset,
    ACTIONS(388), 1,
      anon_sym_COLONname,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
  [3731] = 2,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
    ACTIONS(390), 3,
      ts_builtin_sym_end,
      sym_lparen,
      sym_rparen,
  [3741] = 2,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
    ACTIONS(392), 3,
      ts_builtin_sym_end,
      sym_lparen,
      sym_rparen,
  [3751] = 2,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
    ACTIONS(394), 3,
      ts_builtin_sym_end,
      sym_lparen,
      sym_rparen,
  [3761] = 2,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
    ACTIONS(168), 3,
      sym_lparen,
      sym_rparen,
      sym_ident,
  [3771] = 2,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
    ACTIONS(396), 3,
      ts_builtin_sym_end,
      sym_lparen,
      sym_rparen,
  [3781] = 2,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
    ACTIONS(172), 3,
      sym_lparen,
      sym_rparen,
      sym_ident,
  [3791] = 2,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
    ACTIONS(398), 3,
      ts_builtin_sym_end,
      sym_lparen,
      sym_rparen,
  [3801] = 4,
    ACTIONS(384), 1,
      sym_rparen,
    ACTIONS(400), 1,
      anon_sym_COLONmerge,
    ACTIONS(402), 1,
      anon_sym_COLONdefault,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
  [3815] = 2,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
    ACTIONS(404), 3,
      ts_builtin_sym_end,
      sym_lparen,
      sym_rparen,
  [3825] = 2,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
    ACTIONS(406), 3,
      ts_builtin_sym_end,
      sym_lparen,
      sym_rparen,
  [3835] = 2,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
    ACTIONS(408), 3,
      ts_builtin_sym_end,
      sym_lparen,
      sym_rparen,
  [3845] = 4,
    ACTIONS(273), 1,
      sym_rparen,
    ACTIONS(410), 1,
      anon_sym_COLONuntil,
    ACTIONS(412), 1,
      sym_ident,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
  [3859] = 4,
    ACTIONS(248), 1,
      sym_rparen,
    ACTIONS(414), 1,
      anon_sym_COLONuntil,
    ACTIONS(416), 1,
      sym_ident,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
  [3873] = 4,
    ACTIONS(122), 1,
      sym_rparen,
    ACTIONS(418), 1,
      anon_sym_COLONruleset,
    ACTIONS(420), 1,
      anon_sym_COLONname,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
  [3887] = 2,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
    ACTIONS(422), 3,
      ts_builtin_sym_end,
      sym_lparen,
      sym_rparen,
  [3897] = 2,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
    ACTIONS(424), 3,
      ts_builtin_sym_end,
      sym_lparen,
      sym_rparen,
  [3907] = 4,
    ACTIONS(116), 1,
      sym_rparen,
    ACTIONS(426), 1,
      anon_sym_COLONruleset,
    ACTIONS(428), 1,
      anon_sym_COLONwhen,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
  [3921] = 2,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
    ACTIONS(430), 3,
      ts_builtin_sym_end,
      sym_lparen,
      sym_rparen,
  [3931] = 4,
    ACTIONS(432), 1,
      sym_rparen,
    ACTIONS(434), 1,
      sym_type,
    STATE(130), 1,
      aux_sym_command_repeat4,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
  [3945] = 4,
    ACTIONS(436), 1,
      sym_rparen,
    ACTIONS(438), 1,
      sym_type,
    STATE(125), 1,
      aux_sym_command_repeat4,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
  [3959] = 3,
    ACTIONS(384), 1,
      sym_rparen,
    ACTIONS(440), 1,
      anon_sym_COLONruleset,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
  [3970] = 2,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
    ACTIONS(442), 2,
      sym_lparen,
      sym_rparen,
  [3979] = 3,
    ACTIONS(289), 1,
      sym_rparen,
    ACTIONS(444), 1,
      sym_unum,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
  [3990] = 3,
    ACTIONS(122), 1,
      sym_rparen,
    ACTIONS(309), 1,
      anon_sym_COLONdefault,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
  [4001] = 3,
    ACTIONS(289), 1,
      sym_rparen,
    ACTIONS(444), 1,
      sym_ident,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
  [4012] = 2,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
    ACTIONS(446), 2,
      sym_lparen,
      sym_rparen,
  [4021] = 3,
    ACTIONS(116), 1,
      sym_rparen,
    ACTIONS(448), 1,
      anon_sym_COLONuntil,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
  [4032] = 3,
    ACTIONS(287), 1,
      sym_rparen,
    ACTIONS(450), 1,
      anon_sym_COLONuntil,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
  [4043] = 3,
    ACTIONS(452), 1,
      sym_rparen,
    ACTIONS(454), 1,
      anon_sym_COLONdefault,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
  [4054] = 3,
    ACTIONS(252), 1,
      sym_rparen,
    ACTIONS(456), 1,
      anon_sym_COLONuntil,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
  [4065] = 3,
    ACTIONS(458), 1,
      sym_rparen,
    ACTIONS(460), 1,
      anon_sym_COLONdefault,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
  [4076] = 3,
    ACTIONS(358), 1,
      sym_rparen,
    ACTIONS(462), 1,
      anon_sym_COLONname,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
  [4087] = 2,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
    ACTIONS(464), 2,
      sym_lparen,
      sym_rparen,
  [4096] = 2,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
    ACTIONS(466), 2,
      sym_lparen,
      sym_rparen,
  [4105] = 3,
    ACTIONS(358), 1,
      sym_rparen,
    ACTIONS(362), 1,
      anon_sym_COLONdefault,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
  [4116] = 3,
    ACTIONS(370), 1,
      sym_rparen,
    ACTIONS(468), 1,
      anon_sym_COLONname,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
  [4127] = 3,
    ACTIONS(234), 1,
      sym_rparen,
    ACTIONS(470), 1,
      sym_lparen,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
  [4138] = 3,
    ACTIONS(370), 1,
      sym_rparen,
    ACTIONS(374), 1,
      anon_sym_COLONdefault,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
  [4149] = 3,
    ACTIONS(384), 1,
      sym_rparen,
    ACTIONS(402), 1,
      anon_sym_COLONdefault,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
  [4160] = 3,
    ACTIONS(472), 1,
      sym_lparen,
    STATE(63), 1,
      sym_schema,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
  [4171] = 2,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
    ACTIONS(474), 2,
      sym_lparen,
      sym_rparen,
  [4180] = 3,
    ACTIONS(476), 1,
      sym_rparen,
    ACTIONS(478), 1,
      sym_ident,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
  [4191] = 2,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
    ACTIONS(480), 2,
      sym_lparen,
      sym_rparen,
  [4200] = 3,
    ACTIONS(340), 1,
      sym_rparen,
    ACTIONS(344), 1,
      anon_sym_COLONdefault,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
  [4211] = 3,
    ACTIONS(482), 1,
      sym_unum,
    ACTIONS(484), 1,
      sym_ident,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
  [4222] = 3,
    ACTIONS(340), 1,
      sym_rparen,
    ACTIONS(486), 1,
      anon_sym_COLONruleset,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
  [4233] = 2,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
    ACTIONS(488), 2,
      sym_lparen,
      sym_rparen,
  [4242] = 3,
    ACTIONS(340), 1,
      sym_rparen,
    ACTIONS(382), 1,
      anon_sym_COLONname,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
  [4253] = 3,
    ACTIONS(65), 1,
      sym_ident,
    ACTIONS(490), 1,
      anon_sym_EQ,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
  [4264] = 3,
    ACTIONS(234), 1,
      sym_rparen,
    ACTIONS(492), 1,
      anon_sym_COLONuntil,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
  [4275] = 2,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
    ACTIONS(494), 2,
      sym_lparen,
      sym_rparen,
  [4284] = 2,
    ACTIONS(496), 1,
      sym_type,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
  [4292] = 2,
    ACTIONS(498), 1,
      sym_ident,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
  [4300] = 2,
    ACTIONS(500), 1,
      sym_lparen,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
  [4308] = 2,
    ACTIONS(496), 1,
      sym_unum,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
  [4316] = 2,
    ACTIONS(502), 1,
      sym_ident,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
  [4324] = 2,
    ACTIONS(504), 1,
      sym_ident,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
  [4332] = 2,
    ACTIONS(506), 1,
      sym_string,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
  [4340] = 2,
    ACTIONS(508), 1,
      sym_lparen,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
  [4348] = 2,
    ACTIONS(510), 1,
      sym_unum,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
  [4356] = 2,
    ACTIONS(444), 1,
      sym_ident,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
  [4364] = 2,
    ACTIONS(512), 1,
      sym_lparen,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
  [4372] = 2,
    ACTIONS(514), 1,
      sym_type,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
  [4380] = 2,
    ACTIONS(275), 1,
      sym_rparen,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
  [4388] = 2,
    ACTIONS(516), 1,
      sym_unum,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
  [4396] = 2,
    ACTIONS(340), 1,
      sym_rparen,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
  [4404] = 2,
    ACTIONS(234), 1,
      sym_rparen,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
  [4412] = 2,
    ACTIONS(518), 1,
      sym_ident,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
  [4420] = 2,
    ACTIONS(520), 1,
      sym_string,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
  [4428] = 2,
    ACTIONS(520), 1,
      sym_ident,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
  [4436] = 2,
    ACTIONS(522), 1,
      sym_lparen,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
  [4444] = 2,
    ACTIONS(524), 1,
      sym_rparen,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
  [4452] = 2,
    ACTIONS(526), 1,
      ts_builtin_sym_end,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
  [4460] = 2,
    ACTIONS(528), 1,
      sym_ident,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
  [4468] = 2,
    ACTIONS(530), 1,
      sym_rparen,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
  [4476] = 2,
    ACTIONS(370), 1,
      sym_rparen,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
  [4484] = 2,
    ACTIONS(532), 1,
      sym_ident,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
  [4492] = 2,
    ACTIONS(126), 1,
      sym_rparen,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
  [4500] = 2,
    ACTIONS(534), 1,
      sym_rparen,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
  [4508] = 2,
    ACTIONS(536), 1,
      sym_string,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
  [4516] = 2,
    ACTIONS(116), 1,
      sym_rparen,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
  [4524] = 2,
    ACTIONS(538), 1,
      sym_ident,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
  [4532] = 2,
    ACTIONS(536), 1,
      sym_ident,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
  [4540] = 2,
    ACTIONS(384), 1,
      sym_rparen,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
  [4548] = 2,
    ACTIONS(358), 1,
      sym_rparen,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
  [4556] = 2,
    ACTIONS(540), 1,
      anon_sym_set,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
  [4564] = 2,
    ACTIONS(542), 1,
      sym_ident,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
  [4572] = 2,
    ACTIONS(544), 1,
      sym_ident,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
  [4580] = 2,
    ACTIONS(496), 1,
      sym_string,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
  [4588] = 2,
    ACTIONS(546), 1,
      sym_rparen,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
  [4596] = 2,
    ACTIONS(548), 1,
      sym_string,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
  [4604] = 2,
    ACTIONS(550), 1,
      sym_rparen,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
  [4612] = 2,
    ACTIONS(552), 1,
      sym_type,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
  [4620] = 2,
    ACTIONS(554), 1,
      sym_string,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
  [4628] = 2,
    ACTIONS(556), 1,
      sym_ident,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
  [4636] = 2,
    ACTIONS(458), 1,
      sym_rparen,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
  [4644] = 2,
    ACTIONS(558), 1,
      sym_lparen,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
  [4652] = 2,
    ACTIONS(560), 1,
      sym_string,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
  [4660] = 2,
    ACTIONS(452), 1,
      sym_rparen,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
  [4668] = 2,
    ACTIONS(562), 1,
      sym_rparen,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
  [4676] = 2,
    ACTIONS(444), 1,
      sym_string,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
  [4684] = 2,
    ACTIONS(122), 1,
      sym_rparen,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
  [4692] = 2,
    ACTIONS(564), 1,
      sym_lparen,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
  [4700] = 2,
    ACTIONS(566), 1,
      sym_rparen,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
  [4708] = 2,
    ACTIONS(89), 1,
      sym_ident,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
  [4716] = 2,
    ACTIONS(568), 1,
      sym_lparen,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
  [4724] = 2,
    ACTIONS(570), 1,
      sym_rparen,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
  [4732] = 2,
    ACTIONS(87), 1,
      sym_rparen,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
  [4740] = 2,
    ACTIONS(572), 1,
      sym_unum,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
  [4748] = 2,
    ACTIONS(574), 1,
      sym_lparen,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
  [4756] = 2,
    ACTIONS(576), 1,
      sym_type,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
  [4764] = 2,
    ACTIONS(578), 1,
      sym_ident,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
  [4772] = 2,
    ACTIONS(580), 1,
      sym_ident,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
  [4780] = 2,
    ACTIONS(582), 1,
      sym_ident,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
  [4788] = 2,
    ACTIONS(584), 1,
      sym_string,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
  [4796] = 2,
    ACTIONS(586), 1,
      sym_ident,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
  [4804] = 2,
    ACTIONS(588), 1,
      sym_ident,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
  [4812] = 2,
    ACTIONS(590), 1,
      sym_lparen,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
  [4820] = 2,
    ACTIONS(592), 1,
      sym_ident,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
  [4828] = 2,
    ACTIONS(594), 1,
      sym_unum,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
  [4836] = 2,
    ACTIONS(289), 1,
      sym_rparen,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
  [4844] = 2,
    ACTIONS(596), 1,
      sym_lparen,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
  [4852] = 2,
    ACTIONS(598), 1,
      sym_rparen,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
  [4860] = 2,
    ACTIONS(600), 1,
      sym_ident,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
  [4868] = 2,
    ACTIONS(602), 1,
      sym_unum,
    ACTIONS(3), 2,
      sym_comment,
      sym_ws,
};

static const uint32_t ts_small_parse_table_map[] = {
  [SMALL_STATE(2)] = 0,
  [SMALL_STATE(3)] = 96,
  [SMALL_STATE(4)] = 142,
  [SMALL_STATE(5)] = 188,
  [SMALL_STATE(6)] = 234,
  [SMALL_STATE(7)] = 280,
  [SMALL_STATE(8)] = 326,
  [SMALL_STATE(9)] = 372,
  [SMALL_STATE(10)] = 418,
  [SMALL_STATE(11)] = 464,
  [SMALL_STATE(12)] = 510,
  [SMALL_STATE(13)] = 556,
  [SMALL_STATE(14)] = 602,
  [SMALL_STATE(15)] = 648,
  [SMALL_STATE(16)] = 693,
  [SMALL_STATE(17)] = 736,
  [SMALL_STATE(18)] = 781,
  [SMALL_STATE(19)] = 824,
  [SMALL_STATE(20)] = 867,
  [SMALL_STATE(21)] = 910,
  [SMALL_STATE(22)] = 955,
  [SMALL_STATE(23)] = 997,
  [SMALL_STATE(24)] = 1039,
  [SMALL_STATE(25)] = 1081,
  [SMALL_STATE(26)] = 1123,
  [SMALL_STATE(27)] = 1165,
  [SMALL_STATE(28)] = 1207,
  [SMALL_STATE(29)] = 1249,
  [SMALL_STATE(30)] = 1291,
  [SMALL_STATE(31)] = 1333,
  [SMALL_STATE(32)] = 1375,
  [SMALL_STATE(33)] = 1417,
  [SMALL_STATE(34)] = 1459,
  [SMALL_STATE(35)] = 1501,
  [SMALL_STATE(36)] = 1543,
  [SMALL_STATE(37)] = 1585,
  [SMALL_STATE(38)] = 1627,
  [SMALL_STATE(39)] = 1669,
  [SMALL_STATE(40)] = 1711,
  [SMALL_STATE(41)] = 1753,
  [SMALL_STATE(42)] = 1795,
  [SMALL_STATE(43)] = 1837,
  [SMALL_STATE(44)] = 1879,
  [SMALL_STATE(45)] = 1921,
  [SMALL_STATE(46)] = 1963,
  [SMALL_STATE(47)] = 2005,
  [SMALL_STATE(48)] = 2047,
  [SMALL_STATE(49)] = 2071,
  [SMALL_STATE(50)] = 2095,
  [SMALL_STATE(51)] = 2118,
  [SMALL_STATE(52)] = 2141,
  [SMALL_STATE(53)] = 2164,
  [SMALL_STATE(54)] = 2187,
  [SMALL_STATE(55)] = 2210,
  [SMALL_STATE(56)] = 2233,
  [SMALL_STATE(57)] = 2255,
  [SMALL_STATE(58)] = 2274,
  [SMALL_STATE(59)] = 2293,
  [SMALL_STATE(60)] = 2312,
  [SMALL_STATE(61)] = 2331,
  [SMALL_STATE(62)] = 2350,
  [SMALL_STATE(63)] = 2376,
  [SMALL_STATE(64)] = 2402,
  [SMALL_STATE(65)] = 2423,
  [SMALL_STATE(66)] = 2444,
  [SMALL_STATE(67)] = 2465,
  [SMALL_STATE(68)] = 2478,
  [SMALL_STATE(69)] = 2499,
  [SMALL_STATE(70)] = 2520,
  [SMALL_STATE(71)] = 2541,
  [SMALL_STATE(72)] = 2562,
  [SMALL_STATE(73)] = 2575,
  [SMALL_STATE(74)] = 2596,
  [SMALL_STATE(75)] = 2617,
  [SMALL_STATE(76)] = 2638,
  [SMALL_STATE(77)] = 2659,
  [SMALL_STATE(78)] = 2680,
  [SMALL_STATE(79)] = 2701,
  [SMALL_STATE(80)] = 2719,
  [SMALL_STATE(81)] = 2737,
  [SMALL_STATE(82)] = 2755,
  [SMALL_STATE(83)] = 2773,
  [SMALL_STATE(84)] = 2791,
  [SMALL_STATE(85)] = 2809,
  [SMALL_STATE(86)] = 2827,
  [SMALL_STATE(87)] = 2845,
  [SMALL_STATE(88)] = 2863,
  [SMALL_STATE(89)] = 2875,
  [SMALL_STATE(90)] = 2893,
  [SMALL_STATE(91)] = 2911,
  [SMALL_STATE(92)] = 2929,
  [SMALL_STATE(93)] = 2947,
  [SMALL_STATE(94)] = 2965,
  [SMALL_STATE(95)] = 2983,
  [SMALL_STATE(96)] = 3001,
  [SMALL_STATE(97)] = 3019,
  [SMALL_STATE(98)] = 3037,
  [SMALL_STATE(99)] = 3055,
  [SMALL_STATE(100)] = 3075,
  [SMALL_STATE(101)] = 3095,
  [SMALL_STATE(102)] = 3113,
  [SMALL_STATE(103)] = 3131,
  [SMALL_STATE(104)] = 3149,
  [SMALL_STATE(105)] = 3167,
  [SMALL_STATE(106)] = 3185,
  [SMALL_STATE(107)] = 3203,
  [SMALL_STATE(108)] = 3223,
  [SMALL_STATE(109)] = 3241,
  [SMALL_STATE(110)] = 3259,
  [SMALL_STATE(111)] = 3277,
  [SMALL_STATE(112)] = 3295,
  [SMALL_STATE(113)] = 3313,
  [SMALL_STATE(114)] = 3328,
  [SMALL_STATE(115)] = 3345,
  [SMALL_STATE(116)] = 3360,
  [SMALL_STATE(117)] = 3377,
  [SMALL_STATE(118)] = 3392,
  [SMALL_STATE(119)] = 3407,
  [SMALL_STATE(120)] = 3422,
  [SMALL_STATE(121)] = 3439,
  [SMALL_STATE(122)] = 3454,
  [SMALL_STATE(123)] = 3469,
  [SMALL_STATE(124)] = 3484,
  [SMALL_STATE(125)] = 3499,
  [SMALL_STATE(126)] = 3513,
  [SMALL_STATE(127)] = 3527,
  [SMALL_STATE(128)] = 3537,
  [SMALL_STATE(129)] = 3547,
  [SMALL_STATE(130)] = 3557,
  [SMALL_STATE(131)] = 3571,
  [SMALL_STATE(132)] = 3581,
  [SMALL_STATE(133)] = 3591,
  [SMALL_STATE(134)] = 3601,
  [SMALL_STATE(135)] = 3611,
  [SMALL_STATE(136)] = 3625,
  [SMALL_STATE(137)] = 3639,
  [SMALL_STATE(138)] = 3649,
  [SMALL_STATE(139)] = 3659,
  [SMALL_STATE(140)] = 3673,
  [SMALL_STATE(141)] = 3683,
  [SMALL_STATE(142)] = 3693,
  [SMALL_STATE(143)] = 3707,
  [SMALL_STATE(144)] = 3717,
  [SMALL_STATE(145)] = 3731,
  [SMALL_STATE(146)] = 3741,
  [SMALL_STATE(147)] = 3751,
  [SMALL_STATE(148)] = 3761,
  [SMALL_STATE(149)] = 3771,
  [SMALL_STATE(150)] = 3781,
  [SMALL_STATE(151)] = 3791,
  [SMALL_STATE(152)] = 3801,
  [SMALL_STATE(153)] = 3815,
  [SMALL_STATE(154)] = 3825,
  [SMALL_STATE(155)] = 3835,
  [SMALL_STATE(156)] = 3845,
  [SMALL_STATE(157)] = 3859,
  [SMALL_STATE(158)] = 3873,
  [SMALL_STATE(159)] = 3887,
  [SMALL_STATE(160)] = 3897,
  [SMALL_STATE(161)] = 3907,
  [SMALL_STATE(162)] = 3921,
  [SMALL_STATE(163)] = 3931,
  [SMALL_STATE(164)] = 3945,
  [SMALL_STATE(165)] = 3959,
  [SMALL_STATE(166)] = 3970,
  [SMALL_STATE(167)] = 3979,
  [SMALL_STATE(168)] = 3990,
  [SMALL_STATE(169)] = 4001,
  [SMALL_STATE(170)] = 4012,
  [SMALL_STATE(171)] = 4021,
  [SMALL_STATE(172)] = 4032,
  [SMALL_STATE(173)] = 4043,
  [SMALL_STATE(174)] = 4054,
  [SMALL_STATE(175)] = 4065,
  [SMALL_STATE(176)] = 4076,
  [SMALL_STATE(177)] = 4087,
  [SMALL_STATE(178)] = 4096,
  [SMALL_STATE(179)] = 4105,
  [SMALL_STATE(180)] = 4116,
  [SMALL_STATE(181)] = 4127,
  [SMALL_STATE(182)] = 4138,
  [SMALL_STATE(183)] = 4149,
  [SMALL_STATE(184)] = 4160,
  [SMALL_STATE(185)] = 4171,
  [SMALL_STATE(186)] = 4180,
  [SMALL_STATE(187)] = 4191,
  [SMALL_STATE(188)] = 4200,
  [SMALL_STATE(189)] = 4211,
  [SMALL_STATE(190)] = 4222,
  [SMALL_STATE(191)] = 4233,
  [SMALL_STATE(192)] = 4242,
  [SMALL_STATE(193)] = 4253,
  [SMALL_STATE(194)] = 4264,
  [SMALL_STATE(195)] = 4275,
  [SMALL_STATE(196)] = 4284,
  [SMALL_STATE(197)] = 4292,
  [SMALL_STATE(198)] = 4300,
  [SMALL_STATE(199)] = 4308,
  [SMALL_STATE(200)] = 4316,
  [SMALL_STATE(201)] = 4324,
  [SMALL_STATE(202)] = 4332,
  [SMALL_STATE(203)] = 4340,
  [SMALL_STATE(204)] = 4348,
  [SMALL_STATE(205)] = 4356,
  [SMALL_STATE(206)] = 4364,
  [SMALL_STATE(207)] = 4372,
  [SMALL_STATE(208)] = 4380,
  [SMALL_STATE(209)] = 4388,
  [SMALL_STATE(210)] = 4396,
  [SMALL_STATE(211)] = 4404,
  [SMALL_STATE(212)] = 4412,
  [SMALL_STATE(213)] = 4420,
  [SMALL_STATE(214)] = 4428,
  [SMALL_STATE(215)] = 4436,
  [SMALL_STATE(216)] = 4444,
  [SMALL_STATE(217)] = 4452,
  [SMALL_STATE(218)] = 4460,
  [SMALL_STATE(219)] = 4468,
  [SMALL_STATE(220)] = 4476,
  [SMALL_STATE(221)] = 4484,
  [SMALL_STATE(222)] = 4492,
  [SMALL_STATE(223)] = 4500,
  [SMALL_STATE(224)] = 4508,
  [SMALL_STATE(225)] = 4516,
  [SMALL_STATE(226)] = 4524,
  [SMALL_STATE(227)] = 4532,
  [SMALL_STATE(228)] = 4540,
  [SMALL_STATE(229)] = 4548,
  [SMALL_STATE(230)] = 4556,
  [SMALL_STATE(231)] = 4564,
  [SMALL_STATE(232)] = 4572,
  [SMALL_STATE(233)] = 4580,
  [SMALL_STATE(234)] = 4588,
  [SMALL_STATE(235)] = 4596,
  [SMALL_STATE(236)] = 4604,
  [SMALL_STATE(237)] = 4612,
  [SMALL_STATE(238)] = 4620,
  [SMALL_STATE(239)] = 4628,
  [SMALL_STATE(240)] = 4636,
  [SMALL_STATE(241)] = 4644,
  [SMALL_STATE(242)] = 4652,
  [SMALL_STATE(243)] = 4660,
  [SMALL_STATE(244)] = 4668,
  [SMALL_STATE(245)] = 4676,
  [SMALL_STATE(246)] = 4684,
  [SMALL_STATE(247)] = 4692,
  [SMALL_STATE(248)] = 4700,
  [SMALL_STATE(249)] = 4708,
  [SMALL_STATE(250)] = 4716,
  [SMALL_STATE(251)] = 4724,
  [SMALL_STATE(252)] = 4732,
  [SMALL_STATE(253)] = 4740,
  [SMALL_STATE(254)] = 4748,
  [SMALL_STATE(255)] = 4756,
  [SMALL_STATE(256)] = 4764,
  [SMALL_STATE(257)] = 4772,
  [SMALL_STATE(258)] = 4780,
  [SMALL_STATE(259)] = 4788,
  [SMALL_STATE(260)] = 4796,
  [SMALL_STATE(261)] = 4804,
  [SMALL_STATE(262)] = 4812,
  [SMALL_STATE(263)] = 4820,
  [SMALL_STATE(264)] = 4828,
  [SMALL_STATE(265)] = 4836,
  [SMALL_STATE(266)] = 4844,
  [SMALL_STATE(267)] = 4852,
  [SMALL_STATE(268)] = 4860,
  [SMALL_STATE(269)] = 4868,
};

static const TSParseActionEntry ts_parse_actions[] = {
  [0] = {.entry = {.count = 0, .reusable = false}},
  [1] = {.entry = {.count = 1, .reusable = false}}, RECOVER(),
  [3] = {.entry = {.count = 1, .reusable = true}}, SHIFT_EXTRA(),
  [5] = {.entry = {.count = 1, .reusable = true}}, REDUCE(sym_source_file, 0),
  [7] = {.entry = {.count = 1, .reusable = true}}, SHIFT(2),
  [9] = {.entry = {.count = 1, .reusable = true}}, SHIFT(230),
  [11] = {.entry = {.count = 1, .reusable = false}}, SHIFT(231),
  [13] = {.entry = {.count = 1, .reusable = false}}, SHIFT(197),
  [15] = {.entry = {.count = 1, .reusable = false}}, SHIFT(256),
  [17] = {.entry = {.count = 1, .reusable = false}}, SHIFT(257),
  [19] = {.entry = {.count = 1, .reusable = false}}, SHIFT(258),
  [21] = {.entry = {.count = 1, .reusable = false}}, SHIFT(268),
  [23] = {.entry = {.count = 1, .reusable = false}}, SHIFT(205),
  [25] = {.entry = {.count = 1, .reusable = false}}, SHIFT(203),
  [27] = {.entry = {.count = 1, .reusable = false}}, SHIFT(28),
  [29] = {.entry = {.count = 1, .reusable = false}}, SHIFT(189),
  [31] = {.entry = {.count = 1, .reusable = false}}, SHIFT(136),
  [33] = {.entry = {.count = 1, .reusable = false}}, SHIFT(266),
  [35] = {.entry = {.count = 1, .reusable = false}}, SHIFT(21),
  [37] = {.entry = {.count = 1, .reusable = false}}, SHIFT(109),
  [39] = {.entry = {.count = 1, .reusable = false}}, SHIFT(265),
  [41] = {.entry = {.count = 1, .reusable = false}}, SHIFT(106),
  [43] = {.entry = {.count = 1, .reusable = false}}, SHIFT(167),
  [45] = {.entry = {.count = 1, .reusable = false}}, SHIFT(263),
  [47] = {.entry = {.count = 1, .reusable = false}}, SHIFT(169),
  [49] = {.entry = {.count = 1, .reusable = false}}, SHIFT(260),
  [51] = {.entry = {.count = 1, .reusable = false}}, SHIFT(259),
  [53] = {.entry = {.count = 1, .reusable = false}}, SHIFT(120),
  [55] = {.entry = {.count = 1, .reusable = false}}, SHIFT(245),
  [57] = {.entry = {.count = 1, .reusable = false}}, SHIFT(241),
  [59] = {.entry = {.count = 1, .reusable = false}}, SHIFT(38),
  [61] = {.entry = {.count = 1, .reusable = false}}, SHIFT(238),
  [63] = {.entry = {.count = 1, .reusable = false}}, SHIFT(40),
  [65] = {.entry = {.count = 1, .reusable = false}}, SHIFT(11),
  [67] = {.entry = {.count = 1, .reusable = true}}, SHIFT(186),
  [69] = {.entry = {.count = 1, .reusable = true}}, SHIFT(27),
  [71] = {.entry = {.count = 1, .reusable = false}}, SHIFT(51),
  [73] = {.entry = {.count = 1, .reusable = false}}, SHIFT(52),
  [75] = {.entry = {.count = 1, .reusable = false}}, SHIFT(53),
  [77] = {.entry = {.count = 1, .reusable = true}}, SHIFT(53),
  [79] = {.entry = {.count = 1, .reusable = false}}, SHIFT(54),
  [81] = {.entry = {.count = 1, .reusable = true}}, SHIFT(55),
  [83] = {.entry = {.count = 1, .reusable = true}}, SHIFT(228),
  [85] = {.entry = {.count = 1, .reusable = true}}, SHIFT(216),
  [87] = {.entry = {.count = 1, .reusable = true}}, SHIFT(149),
  [89] = {.entry = {.count = 1, .reusable = true}}, SHIFT(246),
  [91] = {.entry = {.count = 2, .reusable = true}}, REDUCE(aux_sym_command_repeat2, 2), SHIFT_REPEAT(186),
  [94] = {.entry = {.count = 1, .reusable = true}}, REDUCE(aux_sym_command_repeat2, 2),
  [96] = {.entry = {.count = 2, .reusable = false}}, REDUCE(aux_sym_command_repeat2, 2), SHIFT_REPEAT(51),
  [99] = {.entry = {.count = 2, .reusable = false}}, REDUCE(aux_sym_command_repeat2, 2), SHIFT_REPEAT(52),
  [102] = {.entry = {.count = 2, .reusable = false}}, REDUCE(aux_sym_command_repeat2, 2), SHIFT_REPEAT(53),
  [105] = {.entry = {.count = 2, .reusable = true}}, REDUCE(aux_sym_command_repeat2, 2), SHIFT_REPEAT(53),
  [108] = {.entry = {.count = 2, .reusable = false}}, REDUCE(aux_sym_command_repeat2, 2), SHIFT_REPEAT(54),
  [111] = {.entry = {.count = 2, .reusable = true}}, REDUCE(aux_sym_command_repeat2, 2), SHIFT_REPEAT(55),
  [114] = {.entry = {.count = 1, .reusable = true}}, SHIFT(219),
  [116] = {.entry = {.count = 1, .reusable = true}}, SHIFT(145),
  [118] = {.entry = {.count = 1, .reusable = true}}, SHIFT(49),
  [120] = {.entry = {.count = 1, .reusable = true}}, SHIFT(30),
  [122] = {.entry = {.count = 1, .reusable = true}}, SHIFT(159),
  [124] = {.entry = {.count = 1, .reusable = true}}, SHIFT(48),
  [126] = {.entry = {.count = 1, .reusable = true}}, SHIFT(151),
  [128] = {.entry = {.count = 1, .reusable = true}}, SHIFT(204),
  [130] = {.entry = {.count = 1, .reusable = true}}, REDUCE(sym_callexpr, 4),
  [132] = {.entry = {.count = 1, .reusable = false}}, REDUCE(sym_callexpr, 4),
  [134] = {.entry = {.count = 1, .reusable = true}}, REDUCE(sym_callexpr, 3),
  [136] = {.entry = {.count = 1, .reusable = false}}, REDUCE(sym_callexpr, 3),
  [138] = {.entry = {.count = 1, .reusable = true}}, REDUCE(sym_unit, 2),
  [140] = {.entry = {.count = 1, .reusable = false}}, REDUCE(sym_unit, 2),
  [142] = {.entry = {.count = 1, .reusable = true}}, REDUCE(sym_bool, 1),
  [144] = {.entry = {.count = 1, .reusable = false}}, REDUCE(sym_bool, 1),
  [146] = {.entry = {.count = 1, .reusable = true}}, REDUCE(sym_literal, 1),
  [148] = {.entry = {.count = 1, .reusable = false}}, REDUCE(sym_literal, 1),
  [150] = {.entry = {.count = 1, .reusable = true}}, REDUCE(sym_f64, 1),
  [152] = {.entry = {.count = 1, .reusable = false}}, REDUCE(sym_f64, 1),
  [154] = {.entry = {.count = 1, .reusable = true}}, REDUCE(sym_expr, 1),
  [156] = {.entry = {.count = 1, .reusable = false}}, REDUCE(sym_expr, 1),
  [158] = {.entry = {.count = 1, .reusable = true}}, REDUCE(sym_symstring, 1),
  [160] = {.entry = {.count = 1, .reusable = false}}, REDUCE(sym_symstring, 1),
  [162] = {.entry = {.count = 1, .reusable = true}}, REDUCE(aux_sym_command_repeat2, 1),
  [164] = {.entry = {.count = 1, .reusable = true}}, SHIFT(185),
  [166] = {.entry = {.count = 1, .reusable = false}}, REDUCE(aux_sym_command_repeat2, 1),
  [168] = {.entry = {.count = 1, .reusable = true}}, REDUCE(sym_schedule, 5),
  [170] = {.entry = {.count = 1, .reusable = false}}, REDUCE(sym_schedule, 5),
  [172] = {.entry = {.count = 1, .reusable = true}}, REDUCE(sym_schedule, 6),
  [174] = {.entry = {.count = 1, .reusable = false}}, REDUCE(sym_schedule, 6),
  [176] = {.entry = {.count = 1, .reusable = true}}, REDUCE(sym_schedule, 4),
  [178] = {.entry = {.count = 1, .reusable = false}}, REDUCE(sym_schedule, 4),
  [180] = {.entry = {.count = 1, .reusable = true}}, REDUCE(sym_schedule, 3),
  [182] = {.entry = {.count = 1, .reusable = false}}, REDUCE(sym_schedule, 3),
  [184] = {.entry = {.count = 1, .reusable = true}}, REDUCE(sym_schedule, 1),
  [186] = {.entry = {.count = 1, .reusable = false}}, REDUCE(sym_schedule, 1),
  [188] = {.entry = {.count = 1, .reusable = false}}, SHIFT(200),
  [190] = {.entry = {.count = 1, .reusable = true}}, SHIFT(114),
  [192] = {.entry = {.count = 1, .reusable = true}}, SHIFT(254),
  [194] = {.entry = {.count = 1, .reusable = true}}, SHIFT(33),
  [196] = {.entry = {.count = 1, .reusable = true}}, SHIFT(32),
  [198] = {.entry = {.count = 1, .reusable = true}}, SHIFT(253),
  [200] = {.entry = {.count = 1, .reusable = true}}, REDUCE(sym_source_file, 1),
  [202] = {.entry = {.count = 1, .reusable = true}}, SHIFT(62),
  [204] = {.entry = {.count = 1, .reusable = true}}, SHIFT(142),
  [206] = {.entry = {.count = 2, .reusable = true}}, REDUCE(aux_sym_command_repeat3, 2), SHIFT_REPEAT(62),
  [209] = {.entry = {.count = 1, .reusable = true}}, REDUCE(aux_sym_command_repeat3, 2),
  [211] = {.entry = {.count = 1, .reusable = true}}, REDUCE(sym_schema, 3),
  [213] = {.entry = {.count = 1, .reusable = true}}, SHIFT(152),
  [215] = {.entry = {.count = 1, .reusable = true}}, SHIFT(139),
  [217] = {.entry = {.count = 1, .reusable = true}}, SHIFT(126),
  [219] = {.entry = {.count = 1, .reusable = true}}, REDUCE(sym_schema, 4),
  [221] = {.entry = {.count = 1, .reusable = true}}, REDUCE(aux_sym_source_file_repeat1, 2),
  [223] = {.entry = {.count = 2, .reusable = true}}, REDUCE(aux_sym_source_file_repeat1, 2), SHIFT_REPEAT(2),
  [226] = {.entry = {.count = 1, .reusable = true}}, SHIFT(158),
  [228] = {.entry = {.count = 1, .reusable = true}}, SHIFT(144),
  [230] = {.entry = {.count = 1, .reusable = true}}, SHIFT(135),
  [232] = {.entry = {.count = 1, .reusable = true}}, SHIFT(193),
  [234] = {.entry = {.count = 1, .reusable = true}}, SHIFT(153),
  [236] = {.entry = {.count = 1, .reusable = true}}, SHIFT(115),
  [238] = {.entry = {.count = 1, .reusable = true}}, SHIFT(57),
  [240] = {.entry = {.count = 1, .reusable = true}}, SHIFT(127),
  [242] = {.entry = {.count = 1, .reusable = true}}, SHIFT(165),
  [244] = {.entry = {.count = 1, .reusable = true}}, SHIFT(250),
  [246] = {.entry = {.count = 1, .reusable = true}}, REDUCE(sym_cost, 2),
  [248] = {.entry = {.count = 1, .reusable = true}}, SHIFT(60),
  [250] = {.entry = {.count = 1, .reusable = true}}, SHIFT(150),
  [252] = {.entry = {.count = 1, .reusable = true}}, SHIFT(59),
  [254] = {.entry = {.count = 1, .reusable = true}}, SHIFT(190),
  [256] = {.entry = {.count = 1, .reusable = true}}, SHIFT(58),
  [258] = {.entry = {.count = 1, .reusable = true}}, SHIFT(262),
  [260] = {.entry = {.count = 2, .reusable = true}}, REDUCE(aux_sym_command_repeat5, 2), SHIFT_REPEAT(193),
  [263] = {.entry = {.count = 1, .reusable = true}}, REDUCE(aux_sym_command_repeat5, 2),
  [265] = {.entry = {.count = 2, .reusable = true}}, REDUCE(aux_sym_command_repeat7, 2), SHIFT_REPEAT(115),
  [268] = {.entry = {.count = 1, .reusable = true}}, REDUCE(aux_sym_command_repeat7, 2),
  [270] = {.entry = {.count = 2, .reusable = true}}, REDUCE(aux_sym_command_repeat7, 2), SHIFT_REPEAT(127),
  [273] = {.entry = {.count = 1, .reusable = true}}, SHIFT(138),
  [275] = {.entry = {.count = 1, .reusable = true}}, SHIFT(178),
  [277] = {.entry = {.count = 1, .reusable = true}}, SHIFT(119),
  [279] = {.entry = {.count = 1, .reusable = true}}, SHIFT(116),
  [281] = {.entry = {.count = 1, .reusable = true}}, SHIFT(206),
  [283] = {.entry = {.count = 1, .reusable = true}}, SHIFT(29),
  [285] = {.entry = {.count = 1, .reusable = true}}, SHIFT(26),
  [287] = {.entry = {.count = 1, .reusable = true}}, SHIFT(143),
  [289] = {.entry = {.count = 1, .reusable = true}}, SHIFT(154),
  [291] = {.entry = {.count = 1, .reusable = true}}, SHIFT(195),
  [293] = {.entry = {.count = 1, .reusable = true}}, SHIFT(99),
  [295] = {.entry = {.count = 1, .reusable = true}}, SHIFT(148),
  [297] = {.entry = {.count = 1, .reusable = true}}, SHIFT(239),
  [299] = {.entry = {.count = 1, .reusable = true}}, SHIFT(156),
  [301] = {.entry = {.count = 1, .reusable = true}}, SHIFT(98),
  [303] = {.entry = {.count = 1, .reusable = true}}, SHIFT(269),
  [305] = {.entry = {.count = 1, .reusable = true}}, SHIFT(198),
  [307] = {.entry = {.count = 1, .reusable = true}}, SHIFT(25),
  [309] = {.entry = {.count = 1, .reusable = true}}, SHIFT(24),
  [311] = {.entry = {.count = 1, .reusable = true}}, SHIFT(218),
  [313] = {.entry = {.count = 1, .reusable = true}}, SHIFT(16),
  [315] = {.entry = {.count = 1, .reusable = true}}, SHIFT(19),
  [317] = {.entry = {.count = 1, .reusable = true}}, REDUCE(aux_sym_command_repeat4, 2),
  [319] = {.entry = {.count = 2, .reusable = true}}, REDUCE(aux_sym_command_repeat4, 2), SHIFT_REPEAT(119),
  [322] = {.entry = {.count = 1, .reusable = true}}, SHIFT(157),
  [324] = {.entry = {.count = 1, .reusable = true}}, SHIFT(89),
  [326] = {.entry = {.count = 1, .reusable = true}}, SHIFT(264),
  [328] = {.entry = {.count = 2, .reusable = true}}, REDUCE(aux_sym_command_repeat1, 2), SHIFT_REPEAT(239),
  [331] = {.entry = {.count = 1, .reusable = true}}, REDUCE(aux_sym_command_repeat1, 2),
  [333] = {.entry = {.count = 2, .reusable = true}}, REDUCE(aux_sym_command_repeat6, 2), SHIFT_REPEAT(218),
  [336] = {.entry = {.count = 1, .reusable = true}}, REDUCE(aux_sym_command_repeat6, 2),
  [338] = {.entry = {.count = 1, .reusable = true}}, SHIFT(207),
  [340] = {.entry = {.count = 1, .reusable = true}}, SHIFT(140),
  [342] = {.entry = {.count = 1, .reusable = true}}, SHIFT(46),
  [344] = {.entry = {.count = 1, .reusable = true}}, SHIFT(45),
  [346] = {.entry = {.count = 1, .reusable = true}}, REDUCE(sym_command, 15),
  [348] = {.entry = {.count = 1, .reusable = true}}, REDUCE(sym_command, 14),
  [350] = {.entry = {.count = 1, .reusable = true}}, REDUCE(sym_nonletaction, 1),
  [352] = {.entry = {.count = 1, .reusable = true}}, REDUCE(sym_command, 13),
  [354] = {.entry = {.count = 1, .reusable = true}}, REDUCE(sym_command, 12),
  [356] = {.entry = {.count = 1, .reusable = true}}, REDUCE(sym_command, 11),
  [358] = {.entry = {.count = 1, .reusable = true}}, SHIFT(134),
  [360] = {.entry = {.count = 1, .reusable = true}}, SHIFT(36),
  [362] = {.entry = {.count = 1, .reusable = true}}, SHIFT(37),
  [364] = {.entry = {.count = 1, .reusable = true}}, SHIFT(121),
  [366] = {.entry = {.count = 1, .reusable = true}}, SHIFT(61),
  [368] = {.entry = {.count = 1, .reusable = true}}, REDUCE(sym_command, 10),
  [370] = {.entry = {.count = 1, .reusable = true}}, SHIFT(137),
  [372] = {.entry = {.count = 1, .reusable = true}}, SHIFT(42),
  [374] = {.entry = {.count = 1, .reusable = true}}, SHIFT(39),
  [376] = {.entry = {.count = 1, .reusable = true}}, REDUCE(sym_command, 9),
  [378] = {.entry = {.count = 1, .reusable = true}}, REDUCE(sym_nonletaction, 5),
  [380] = {.entry = {.count = 1, .reusable = true}}, SHIFT(226),
  [382] = {.entry = {.count = 1, .reusable = true}}, SHIFT(224),
  [384] = {.entry = {.count = 1, .reusable = true}}, SHIFT(146),
  [386] = {.entry = {.count = 1, .reusable = true}}, SHIFT(212),
  [388] = {.entry = {.count = 1, .reusable = true}}, SHIFT(213),
  [390] = {.entry = {.count = 1, .reusable = true}}, REDUCE(sym_command, 5),
  [392] = {.entry = {.count = 1, .reusable = true}}, REDUCE(sym_command, 8),
  [394] = {.entry = {.count = 1, .reusable = true}}, REDUCE(sym_nonletaction, 8),
  [396] = {.entry = {.count = 1, .reusable = true}}, REDUCE(sym_command, 6),
  [398] = {.entry = {.count = 1, .reusable = true}}, REDUCE(sym_nonletaction, 4),
  [400] = {.entry = {.count = 1, .reusable = true}}, SHIFT(23),
  [402] = {.entry = {.count = 1, .reusable = true}}, SHIFT(31),
  [404] = {.entry = {.count = 1, .reusable = true}}, REDUCE(sym_command, 4),
  [406] = {.entry = {.count = 1, .reusable = true}}, REDUCE(sym_command, 3),
  [408] = {.entry = {.count = 1, .reusable = true}}, REDUCE(sym_command, 1),
  [410] = {.entry = {.count = 1, .reusable = true}}, SHIFT(102),
  [412] = {.entry = {.count = 1, .reusable = true}}, SHIFT(172),
  [414] = {.entry = {.count = 1, .reusable = true}}, SHIFT(91),
  [416] = {.entry = {.count = 1, .reusable = true}}, SHIFT(174),
  [418] = {.entry = {.count = 1, .reusable = true}}, SHIFT(201),
  [420] = {.entry = {.count = 1, .reusable = true}}, SHIFT(202),
  [422] = {.entry = {.count = 1, .reusable = true}}, REDUCE(sym_command, 7),
  [424] = {.entry = {.count = 1, .reusable = true}}, REDUCE(sym_nonletaction, 6),
  [426] = {.entry = {.count = 1, .reusable = true}}, SHIFT(249),
  [428] = {.entry = {.count = 1, .reusable = true}}, SHIFT(247),
  [430] = {.entry = {.count = 1, .reusable = true}}, REDUCE(sym_nonletaction, 7),
  [432] = {.entry = {.count = 1, .reusable = true}}, SHIFT(252),
  [434] = {.entry = {.count = 1, .reusable = true}}, SHIFT(130),
  [436] = {.entry = {.count = 1, .reusable = true}}, SHIFT(255),
  [438] = {.entry = {.count = 1, .reusable = true}}, SHIFT(125),
  [440] = {.entry = {.count = 1, .reusable = true}}, SHIFT(214),
  [442] = {.entry = {.count = 1, .reusable = true}}, REDUCE(sym_identsort, 4),
  [444] = {.entry = {.count = 1, .reusable = true}}, SHIFT(211),
  [446] = {.entry = {.count = 1, .reusable = true}}, REDUCE(sym_action, 1),
  [448] = {.entry = {.count = 1, .reusable = true}}, SHIFT(93),
  [450] = {.entry = {.count = 1, .reusable = true}}, SHIFT(111),
  [452] = {.entry = {.count = 1, .reusable = true}}, SHIFT(132),
  [454] = {.entry = {.count = 1, .reusable = true}}, SHIFT(34),
  [456] = {.entry = {.count = 1, .reusable = true}}, SHIFT(82),
  [458] = {.entry = {.count = 1, .reusable = true}}, SHIFT(133),
  [460] = {.entry = {.count = 1, .reusable = true}}, SHIFT(35),
  [462] = {.entry = {.count = 1, .reusable = true}}, SHIFT(242),
  [464] = {.entry = {.count = 1, .reusable = true}}, REDUCE(sym_action, 5),
  [466] = {.entry = {.count = 1, .reusable = true}}, REDUCE(sym_variant, 4),
  [468] = {.entry = {.count = 1, .reusable = true}}, SHIFT(235),
  [470] = {.entry = {.count = 1, .reusable = true}}, SHIFT(261),
  [472] = {.entry = {.count = 1, .reusable = true}}, SHIFT(164),
  [474] = {.entry = {.count = 1, .reusable = true}}, REDUCE(sym_fact, 5),
  [476] = {.entry = {.count = 1, .reusable = true}}, SHIFT(50),
  [478] = {.entry = {.count = 1, .reusable = true}}, SHIFT(11),
  [480] = {.entry = {.count = 1, .reusable = true}}, REDUCE(sym_variant, 5),
  [482] = {.entry = {.count = 1, .reusable = true}}, SHIFT(194),
  [484] = {.entry = {.count = 1, .reusable = true}}, SHIFT(209),
  [486] = {.entry = {.count = 1, .reusable = true}}, SHIFT(227),
  [488] = {.entry = {.count = 1, .reusable = true}}, REDUCE(sym_fact, 1),
  [490] = {.entry = {.count = 1, .reusable = false}}, SHIFT(18),
  [492] = {.entry = {.count = 1, .reusable = true}}, SHIFT(86),
  [494] = {.entry = {.count = 1, .reusable = true}}, REDUCE(sym_variant, 3),
  [496] = {.entry = {.count = 1, .reusable = true}}, SHIFT(225),
  [498] = {.entry = {.count = 1, .reusable = true}}, SHIFT(124),
  [500] = {.entry = {.count = 1, .reusable = true}}, SHIFT(73),
  [502] = {.entry = {.count = 1, .reusable = true}}, SHIFT(47),
  [504] = {.entry = {.count = 1, .reusable = true}}, SHIFT(192),
  [506] = {.entry = {.count = 1, .reusable = true}}, SHIFT(210),
  [508] = {.entry = {.count = 1, .reusable = true}}, SHIFT(95),
  [510] = {.entry = {.count = 1, .reusable = true}}, SHIFT(44),
  [512] = {.entry = {.count = 1, .reusable = true}}, SHIFT(71),
  [514] = {.entry = {.count = 1, .reusable = true}}, SHIFT(72),
  [516] = {.entry = {.count = 1, .reusable = true}}, SHIFT(171),
  [518] = {.entry = {.count = 1, .reusable = true}}, SHIFT(180),
  [520] = {.entry = {.count = 1, .reusable = true}}, SHIFT(220),
  [522] = {.entry = {.count = 1, .reusable = true}}, SHIFT(163),
  [524] = {.entry = {.count = 1, .reusable = true}}, SHIFT(162),
  [526] = {.entry = {.count = 1, .reusable = true}},  ACCEPT_INPUT(),
  [528] = {.entry = {.count = 1, .reusable = true}}, SHIFT(237),
  [530] = {.entry = {.count = 1, .reusable = true}}, SHIFT(160),
  [532] = {.entry = {.count = 1, .reusable = true}}, SHIFT(9),
  [534] = {.entry = {.count = 1, .reusable = true}}, SHIFT(177),
  [536] = {.entry = {.count = 1, .reusable = true}}, SHIFT(229),
  [538] = {.entry = {.count = 1, .reusable = true}}, SHIFT(176),
  [540] = {.entry = {.count = 1, .reusable = true}}, SHIFT(232),
  [542] = {.entry = {.count = 1, .reusable = true}}, SHIFT(43),
  [544] = {.entry = {.count = 1, .reusable = true}}, SHIFT(12),
  [546] = {.entry = {.count = 1, .reusable = true}}, SHIFT(187),
  [548] = {.entry = {.count = 1, .reusable = true}}, SHIFT(240),
  [550] = {.entry = {.count = 1, .reusable = true}}, SHIFT(147),
  [552] = {.entry = {.count = 1, .reusable = true}}, SHIFT(244),
  [554] = {.entry = {.count = 1, .reusable = true}}, SHIFT(222),
  [556] = {.entry = {.count = 1, .reusable = true}}, SHIFT(107),
  [558] = {.entry = {.count = 1, .reusable = true}}, SHIFT(221),
  [560] = {.entry = {.count = 1, .reusable = true}}, SHIFT(243),
  [562] = {.entry = {.count = 1, .reusable = true}}, SHIFT(166),
  [564] = {.entry = {.count = 1, .reusable = true}}, SHIFT(81),
  [566] = {.entry = {.count = 1, .reusable = true}}, SHIFT(129),
  [568] = {.entry = {.count = 1, .reusable = true}}, SHIFT(76),
  [570] = {.entry = {.count = 1, .reusable = true}}, SHIFT(128),
  [572] = {.entry = {.count = 1, .reusable = true}}, SHIFT(88),
  [574] = {.entry = {.count = 1, .reusable = true}}, SHIFT(68),
  [576] = {.entry = {.count = 1, .reusable = true}}, SHIFT(67),
  [578] = {.entry = {.count = 1, .reusable = true}}, SHIFT(181),
  [580] = {.entry = {.count = 1, .reusable = true}}, SHIFT(184),
  [582] = {.entry = {.count = 1, .reusable = true}}, SHIFT(196),
  [584] = {.entry = {.count = 1, .reusable = true}}, SHIFT(20),
  [586] = {.entry = {.count = 1, .reusable = true}}, SHIFT(233),
  [588] = {.entry = {.count = 1, .reusable = true}}, SHIFT(7),
  [590] = {.entry = {.count = 1, .reusable = true}}, SHIFT(75),
  [592] = {.entry = {.count = 1, .reusable = true}}, SHIFT(199),
  [594] = {.entry = {.count = 1, .reusable = true}}, SHIFT(103),
  [596] = {.entry = {.count = 1, .reusable = true}}, SHIFT(118),
  [598] = {.entry = {.count = 1, .reusable = true}}, SHIFT(141),
  [600] = {.entry = {.count = 1, .reusable = true}}, SHIFT(215),
  [602] = {.entry = {.count = 1, .reusable = true}}, SHIFT(105),
};

#ifdef __cplusplus
extern "C" {
#endif
#ifdef _WIN32
#define extern __declspec(dllexport)
#endif

extern const TSLanguage *tree_sitter_egglog(void) {
  static const TSLanguage language = {
    .version = LANGUAGE_VERSION,
    .symbol_count = SYMBOL_COUNT,
    .alias_count = ALIAS_COUNT,
    .token_count = TOKEN_COUNT,
    .external_token_count = EXTERNAL_TOKEN_COUNT,
    .state_count = STATE_COUNT,
    .large_state_count = LARGE_STATE_COUNT,
    .production_id_count = PRODUCTION_ID_COUNT,
    .field_count = FIELD_COUNT,
    .max_alias_sequence_length = MAX_ALIAS_SEQUENCE_LENGTH,
    .parse_table = &ts_parse_table[0][0],
    .small_parse_table = ts_small_parse_table,
    .small_parse_table_map = ts_small_parse_table_map,
    .parse_actions = ts_parse_actions,
    .symbol_names = ts_symbol_names,
    .symbol_metadata = ts_symbol_metadata,
    .public_symbol_map = ts_symbol_map,
    .alias_map = ts_non_terminal_alias_map,
    .alias_sequences = &ts_alias_sequences[0][0],
    .lex_modes = ts_lex_modes,
    .lex_fn = ts_lex,
    .primary_state_ids = ts_primary_state_ids,
  };
  return &language;
}
#ifdef __cplusplus
}
#endif
