#!/bin/sh
# Shared argument parser for local model launchers; sourcing starts no service.
splash_parse_launcher_options() {
    kv_format=int8
    use_baseline=0
    while [ "$#" -gt 0 ]; do
        case "$1" in
            --kv-format)
                if [ "$#" -lt 2 ]; then
                    echo "--kv-format requires a value" >&2
                    exit 2
                fi
                kv_format=$2
                shift 2
                ;;
            --kv-format=*) kv_format=${1#*=}; shift ;;
            --baseline)
                if [ "${launcher_allow_baseline:-0}" -ne 1 ]; then
                    echo "--baseline is not supported by this launcher" >&2
                    exit 2
                fi
                use_baseline=1
                shift
                ;;
            --help|-h)
                echo "Usage: $0 [--kv-format int8|bf16|q4_0|q4_0_unrotated|q4h]"
                if [ "${launcher_allow_baseline:-0}" -eq 1 ]; then
                    echo "       $0 --baseline [--kv-format int8|bf16]"
                fi
                exit 0
                ;;
            *) echo "Unknown launcher option: $1 (use --help)" >&2; exit 2 ;;
        esac
    done
    case "$kv_format" in
        int8|bf16|q4_0|q4_0_unrotated|q4h) ;;
        *) echo "--kv-format requires int8, bf16, q4_0, q4_0_unrotated or q4h" >&2; exit 2 ;;
    esac
}
