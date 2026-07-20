package semanticvg_cj

import semanticvg_cj.bench.*
import semanticvg_cj.examples.*

main() {
    println("semanticvg-cj 仓颉工程")
    println("  0 = 合成向量最小示例")
    println("  1 = 数据集评测（读取 precomputed TXT，交互选择）")
    print("请选择 [0/1，默认 1]: ")
    let choice = readln()
    if (choice == "0") {
        println("\n=== 最小示例 ===")
        runMinimalVectorMapDemo()
    } else {
        interactiveBenchMain()
    }
}
