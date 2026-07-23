package semanticvg_cj.examples

import std.collection.*
import std.math.*
import std.sort.*
import semanticvg_cj.storage.*

private func requireTrue(condition: Bool, message: String): Unit {
    if (!condition) { throw Exception("回归检查失败: ${message}") }
}

private func makeClusterFixture(): VectorGroup {
    let vectors = ArrayList<Array<Float64>>()
    let descriptions = ArrayList<Any>()
    for (i in 0..16) {
        // 故意交替排列两个簇；顺序切块无法正确分开。
        let x = if (i % 2 == 0) { -10.0 + Float64(i) * 0.001 } else { 10.0 + Float64(i) * 0.001 }
        vectors.add([x, 0.0])
        descriptions.add("item-${i}")
    }
    return VectorGroup(
        "alternating",
        [0.0, 0.0],
        "fixture representative",
        vectors.toArray(),
        descriptions.toArray(),
        vectorType: "test",
        groupType: "fixture"
    )
}

private func groupMemberTrace(group: VectorGroup): String {
    let members = ArrayList<String>()
    for (description in group.descriptions) {
        let textOpt = description["text"] as String
        if (textOpt.isSome()) { members.add(textOpt.getOrThrow()) }
    }
    sort(members, by: { a, b => a.compare(b) })
    var out = ""
    for (i in 0..members.size) {
        if (i > 0) { out += "," }
        out += members[i]
    }
    return out
}

private func resultMemberTrace(results: Array<SearchResult>): String {
    var out = ""
    for (i in 0..results.size) {
        if (i > 0) { out += "," }
        let result = results[i]
        if (result.vectorIdx.isSome()) {
            let description = result.group.descriptions[result.vectorIdx.getOrThrow()]
            let textOpt = description["text"] as String
            out += if (textOpt.isSome()) { textOpt.getOrThrow() } else { "missing" }
        } else {
            out += result.group.groupName
        }
    }
    return out
}

public func runCoreRegressionChecks(): Unit {
    requireTrue(clusterCount(16, 0.75) == 2, "独立聚类模块的 clusterCount 错误")
    requireTrue(abs(mixedDistance(10.0, Some(2.0), 0.25) - 8.0) < 1e-12,
        "独立 mixed scoring 模块计算错误")
    let vm = VectorMap()
    let created = vm.insertWithAutoCluster("topic", makeClusterFixture(), alpha: 0.75)
    requireTrue(created == 2, "16^(1-0.75) 应产生 2 个簇，实际 ${created}")

    let leftOpt = vm.getGroupByName("topic-0001", "alternating_cluster_0001")
    let rightOpt = vm.getGroupByName("topic-0002", "alternating_cluster_0002")
    requireTrue(leftOpt.isSome() && rightOpt.isSome(), "聚类子组命名或插入失败")
    let left = leftOpt.getOrThrow()
    let right = rightOpt.getOrThrow()
    requireTrue(left.vectors.size == 8 && right.vectors.size == 8, "两个确定性簇应各含 8 个向量")
    requireTrue(abs(left.representative[0] - right.representative[0]) > 15.0, "KMeans 未分开交替排列的两个簇")
    let clusterTraces = ArrayList<String>()
    clusterTraces.add(groupMemberTrace(left))
    clusterTraces.add(groupMemberTrace(right))
    sort(clusterTraces, by: { a, b => a.compare(b) })
    println("TRACE|clusters|${clusterTraces[0]};${clusterTraces[1]}")

    let keyVectors = HashMap<String, Array<Float64>>()
    keyVectors["topic"] = [1.0, 0.0]
    vm.setKeyVectors(keyVectors)
    vm.buildIndex()
    vm.buildHdmgWithConfig(HdmgBuildConfig(embeddingK: 1, semanticIntraK: 1))
    let builtState = vm.getIndexState()
    requireTrue((builtState["rep_index_current"] as Bool).getOrThrow(), "representative 索引应为 current")
    requireTrue((builtState["single_index_current"] as Bool).getOrThrow(), "single 索引应为 current")
    requireTrue((builtState["hdmg_current"] as Bool).getOrThrow(), "HDMG 应为 current")
    requireTrue((builtState["rep_index_mode"] as String).getOrThrow() == "exact_scan_snapshot",
        "rep index 必须明确暴露 exact-scan 语义")
    let buildStats = vm.getLastHdmgSearchStats()
    requireTrue((buildStats["t_build_ns"] as Int64).getOrThrow() >= 0, "HDMG 应记录构建耗时")
    requireTrue((buildStats["embedding_edges"] as Int64).getOrThrow() > 0, "HDMG 应记录 embedding 边数")

    let configuredSearch = vm.searchHdmgWithConfig(
        [-10.0, 0.0], Some([1.0, 0.0]), 0.3, 2,
        HdmgSearchConfig(maxSteps: 10, clusterPoolMultiplier: 2)
    )
    requireTrue(configuredSearch.size == 2, "HDMG 配置对象搜索入口失败")
    let paperAblation = vm.searchWithoutHdmg(
        [-10.0, 0.0], queryKeyVector: Some([1.0, 0.0]), beta: 0.3,
        topK: 2, groupBudget: 1, microClusterBudget: 2
    )
    requireTrue(paperAblation.size == 2, "论文 w/o HDMG 分层直接搜索失败")

    let description = HashMap<String, Any>()
    description["key"] = "topic-0001"
    description["group_name"] = "alternating_cluster_0001"
    description["text"] = "added-item"
    let inserted = vm.addVector([0.0, 1.0], description)
    requireTrue(inserted >= 0, "追加向量失败")
    let staleState = vm.getIndexState()
    requireTrue(!(staleState["hdmg_current"] as Bool).getOrThrow(), "追加向量后 HDMG 必须失效")

    let beforeCrud = left.size()
    requireTrue(vm.insertObject(
        "topic-0001", "alternating_cluster_0001", "stable-object-1", [0.5, 0.5]
    ), "稳定 ID 对象插入失败")
    requireTrue(left.size() == beforeCrud + 1, "对象插入后组大小错误")
    requireTrue(vm.updateObject("stable-object-1", [0.75, 0.25]), "稳定 ID 对象更新失败")
    requireTrue(vm.deleteObject("stable-object-1"), "稳定 ID 对象删除失败")
    requireTrue(left.size() == beforeCrud, "对象删除后组大小未恢复")
    requireTrue(!vm.updateObject("missing-object", [0.0, 0.0]), "不存在的对象不应更新成功")
    requireTrue(!vm.deleteObject("missing-object"), "不存在的对象不应删除成功")

    requireTrue(vm.search([1.0, 0.0], topK: 0).isEmpty(), "topK=0 应返回空结果")
    let betaOne = vm.searchWithMixedKeyRepVec([10.0, 0.0], queryKeyVector: Some([1.0, 0.0]), beta: 1.0, topK: 2)
    let betaClamped = vm.searchWithMixedKeyRepVec([10.0, 0.0], queryKeyVector: Some([1.0, 0.0]), beta: 2.0, topK: 2)
    requireTrue(betaOne.size == betaClamped.size, "beta 限幅后的结果数量不一致")
    for (i in 0..betaOne.size) {
        requireTrue(abs(betaOne[i].distance - betaClamped[i].distance) < 1e-12, "beta>1 未正确限幅")
    }

    let leftResults = vm.search([-10.0, 0.0], topK: 3, mode: "single")
    let rightResults = vm.search([10.0, 0.0], topK: 3, mode: "single")
    println("TRACE|search|left|${resultMemberTrace(leftResults)}")
    println("TRACE|search|right|${resultMemberTrace(rightResults)}")

    var dimensionError = false
    try {
        let _ = cosineDistance([1.0, 0.0], [1.0])
    } catch (e: IllegalArgumentException) {
        dimensionError = true
    }
    requireTrue(dimensionError, "维度不一致必须显式失败，不能返回伪距离 0")

    println("Core regression checks passed.")
}
