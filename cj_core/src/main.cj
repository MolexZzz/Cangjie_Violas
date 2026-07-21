package semanticvg_cj

import std.convert.*
import semanticvg_cj.bench.*
import semanticvg_cj.examples.*

main() {
    println("semanticvg-cj 仓颉工程")
    println("  0 = 合成向量最小示例")
    println("  1 = 数据集评测（读取 precomputed TXT，交互选择）")
    println("  2 = 仓颉核心回归检查")
    println("  bench <smoke|partial|full> <1..6|t|v|a> = 单行批量评测")
    println("  parity <1..6> <maxQueries> <beta> = 真实样例逐 query 对照输出")
    println("  dbbench <mock|milvus|qdrant|chroma> <1..6> <smoke|partial|full> = 外部数据库评测")
    print("请选择: ")
    let choice = readln()
    if (choice == "0") {
        println("\n=== 最小示例 ===")
        runMinimalVectorMapDemo()
    } else if (choice == "2") {
        println("\n=== 核心回归检查 ===")
        runCoreRegressionChecks()
    } else if (choice.startsWith("bench ")) {
        let parts = choice.split(" ")
        if (parts.size != 3) {
            throw IllegalArgumentException("用法: bench <smoke|partial|full> <1..6|t|v|a>")
        }
        runBenchPreset(parts[1], parts[2])
    } else if (choice.startsWith("parity ")) {
        let parts = choice.split(" ")
        if (parts.size != 4) {
            throw IllegalArgumentException("用法: parity <1..6> <maxQueries> <beta>")
        }
        runParityPreset(parts[1], Int64.parse(parts[2]), Float64.parse(parts[3]))
    } else if (choice.startsWith("dbbench ")) {
        let parts = choice.split(" ")
        if (parts.size != 4) {
            throw IllegalArgumentException("用法: dbbench <mock|milvus|qdrant|chroma> <1..6> <smoke|partial|full>")
        }
        runExternalDbPreset(parts[1], parts[2], parts[3])
    } else {
        interactiveBenchMain()
    }
}
