package semanticvg_cj.bench

import std.collection.*
import std.core.*
import std.convert.*
import std.sort.*
import std.time.*
import std.math.*
import semanticvg_cj.storage.*

// 与 Python 20news_bench 默认 TOP_K 对齐
let defaultTopK: Int64 = 3

public class MethodRecallStats {
    public let label: String
    public let recalls: ArrayList<Float64>
    public let ndcgs: ArrayList<Float64>
    public let latencyNs: ArrayList<Int64>
    public var totalLatencyNs: Int64

    public init(label: String) {
        this.label = label
        this.recalls = ArrayList<Float64>()
        this.ndcgs = ArrayList<Float64>()
        this.latencyNs = ArrayList<Int64>()
        this.totalLatencyNs = 0
    }

    public func avg(): Float64 {
        if (recalls.isEmpty()) { return 0.0 }
        var s = 0.0
        for (r in recalls) { s += r }
        return s / Float64(recalls.size)
    }

    public func minVal(): Float64 {
        if (recalls.isEmpty()) { return 0.0 }
        var m = recalls[0]
        for (r in recalls) {
            if (r < m) { m = r }
        }
        return m
    }

    public func maxVal(): Float64 {
        if (recalls.isEmpty()) { return 0.0 }
        var m = recalls[0]
        for (r in recalls) {
            if (r > m) { m = r }
        }
        return m
    }

    public func avgLatencyMs(): Float64 {
        if (latencyNs.isEmpty()) { return 0.0 }
        return Float64(totalLatencyNs) / 1000000.0 / Float64(latencyNs.size)
    }

    public func avgNdcg(): Float64 {
        if (ndcgs.isEmpty()) { return 0.0 }
        var s = 0.0
        for (v in ndcgs) { s += v }
        return s / Float64(ndcgs.size)
    }

    public func addLatency(ns: Int64): Unit {
        latencyNs.add(ns)
        totalLatencyNs += ns
    }

    public func percentileLatencyMs(percentile: Float64): Float64 {
        if (latencyNs.isEmpty()) { return 0.0 }
        let values = ArrayList<Int64>(latencyNs)
        sort(values, by: { a, b =>
            if (a < b) { Ordering.LT } else if (a > b) { Ordering.GT } else { Ordering.EQ }
        })
        let p = if (percentile < 0.0) { 0.0 } else if (percentile > 1.0) { 1.0 } else { percentile }
        let idx = Int64(Float64(values.size - 1) * p + 0.5)
        return Float64(values[idx]) / 1000000.0
    }
}

public class HdmgTraceStats {
    public var queries: Int64
    public var totalDepth: Float64
    public var totalHops: Float64
    public var totalCandidatePool: Float64
    public var totalSelectedNodes: Float64
    public var totalMixedScoreCalls: Float64
    public var totalEntryNs: Int64
    public var totalWalkNs: Int64
    public var totalRerankNs: Int64
    public var totalSearchNs: Int64

    public init() {
        this.queries = 0
        this.totalDepth = 0.0
        this.totalHops = 0.0
        this.totalCandidatePool = 0.0
        this.totalSelectedNodes = 0.0
        this.totalMixedScoreCalls = 0.0
        this.totalEntryNs = 0
        this.totalWalkNs = 0
        this.totalRerankNs = 0
        this.totalSearchNs = 0
    }

    public func avgDepth(): Float64 {
        if (queries == 0) { return 0.0 }
        return totalDepth / Float64(queries)
    }

    public func avgHops(): Float64 {
        if (queries == 0) { return 0.0 }
        return totalHops / Float64(queries)
    }

    public func avgCandidatePool(): Float64 {
        if (queries == 0) { return 0.0 }
        return totalCandidatePool / Float64(queries)
    }

    public func avgSelectedNodes(): Float64 {
        if (queries == 0) { return 0.0 }
        return totalSelectedNodes / Float64(queries)
    }

    public func avgMixedScoreCalls(): Float64 {
        if (queries == 0) { return 0.0 }
        return totalMixedScoreCalls / Float64(queries)
    }

    public func avgEntryMs(): Float64 {
        if (queries == 0) { return 0.0 }
        return Float64(totalEntryNs) / 1000000.0 / Float64(queries)
    }

    public func avgWalkMs(): Float64 {
        if (queries == 0) { return 0.0 }
        return Float64(totalWalkNs) / 1000000.0 / Float64(queries)
    }

    public func avgRerankMs(): Float64 {
        if (queries == 0) { return 0.0 }
        return Float64(totalRerankNs) / 1000000.0 / Float64(queries)
    }

    public func avgSearchMs(): Float64 {
        if (queries == 0) { return 0.0 }
        return Float64(totalSearchNs) / 1000000.0 / Float64(queries)
    }
}

class FlatItem {
    public let id: String
    public let key: String
    public let vector: Array<Float64>

    public init(id: String, key: String, vector: Array<Float64>) {
        this.id = id
        this.key = key
        this.vector = vector
    }
}

class FlatScore <: Comparable<FlatScore> {
    public let score: Float64
    public let id: String
    public let key: String

    public init(score: Float64, id: String, key: String) {
        this.score = score
        this.id = id
        this.key = key
    }

    public func compare(other: FlatScore): Ordering {
        if (this.score < other.score) { return Ordering.LT }
        if (this.score > other.score) { return Ordering.GT }
        let idOrder = this.id.compare(other.id)
        if (idOrder != Ordering.EQ) { return idOrder }
        return this.key.compare(other.key)
    }
}

public class BetaEvalStats {
    public let beta: Float64
    public let mixed: MethodRecallStats
    public let representative: MethodRecallStats
    public let pyflat: MethodRecallStats
    public let hdmgTrace: HdmgTraceStats
    public var totalBuildNs: Int64
    public var buildCount: Int64

    public init(beta: Float64) {
        this.beta = beta
        this.mixed = MethodRecallStats("Mixed")
        this.representative = MethodRecallStats("Representative")
        this.pyflat = MethodRecallStats("ExactSearch")
        this.hdmgTrace = HdmgTraceStats()
        this.totalBuildNs = 0
        this.buildCount = 0
    }

    public func avgBuildMs(): Float64 {
        if (buildCount == 0) { return 0.0 }
        return Float64(totalBuildNs) / 1000000.0 / Float64(buildCount)
    }
}

func padRight(s: String, width: Int64): String {
    let n = Int64(s.size)
    if (n >= width) { return s }
    var out = s
    for (_ in n..width) {
        out = out + " "
    }
    return out
}

func formatFloat4(v: Float64): String {
    let scaled = Int64(v * 10000.0 + 0.5)
    let whole = scaled / 10000
    let frac = scaled % 10000
    let f1 = (frac / 1000) % 10
    let f2 = (frac / 100) % 10
    let f3 = (frac / 10) % 10
    let f4 = frac % 10
    return "${whole}.${f1}${f2}${f3}${f4}"
}

func nowNs(): Int64 {
    return DateTime.now().toUnixTimeStamp().toNanoseconds()
}

func formatBeta(v: Float64): String {
    return formatFloat2(v)
}

func formatFloat2(v: Float64): String {
    let scaled = Int64(v * 100.0 + 0.5)
    let whole = scaled / 100
    let frac = scaled % 100
    let f1 = (frac / 10) % 10
    let f2 = frac % 10
    return "${whole}.${f1}${f2}"
}

func formatMs(v: Float64): String {
    return formatFloat4(v)
}

func recallKeyAtK(results: Array<SearchResult>, trueKey: String, topK: Int64): Float64 {
    let limit = if (Int64(results.size) < topK) { Int64(results.size) } else { topK }
    for (i in 0..limit) {
        if (results[i].key == trueKey) {
            return 1.0
        }
    }
    return 0.0
}

func binaryNdcgAtK(relevant: ArrayList<String>, ranked: ArrayList<String>, topK: Int64): Float64 {
    if (topK <= 0 || relevant.isEmpty()) { return 0.0 }
    let uniqueRelevant = ArrayList<String>()
    for (item in relevant) {
        if (!uniqueRelevant.contains(item)) { uniqueRelevant.add(item) }
    }
    let seen = ArrayList<String>()
    let limit = if (topK < ranked.size) { topK } else { ranked.size }
    var dcg = 0.0
    for (i in 0..limit) {
        let item = ranked[i]
        if (uniqueRelevant.contains(item) && !seen.contains(item)) {
            dcg += 1.0 / (log(Float64(i + 2)) / log(2.0))
            seen.add(item)
        }
    }
    let idealLimit = if (topK < uniqueRelevant.size) { topK } else { uniqueRelevant.size }
    var idcg = 0.0
    for (i in 0..idealLimit) {
        idcg += 1.0 / (log(Float64(i + 2)) / log(2.0))
    }
    return if (idcg > 0.0) { dcg / idcg } else { 0.0 }
}

func keyNdcgAtK(results: Array<SearchResult>, trueKey: String, topK: Int64): Float64 {
    let relevant = ArrayList<String>()
    relevant.add(trueKey)
    return binaryNdcgAtK(relevant, resultBaseKeys(results, topK), topK)
}

func sortedFolderNames(dataset: HashMap<String, FolderData>): ArrayList<String> {
    let names = ArrayList<String>()
    for ((folder, _) in dataset) { names.add(folder) }
    sort(names, by: { a, b => a.compare(b) })
    return names
}

func datasetDimension(dataset: HashMap<String, FolderData>): Int64 {
    for (folder in sortedFolderNames(dataset)) {
        let data = dataset[folder]
        if (!data.vectors.isEmpty()) {
            return Int64(data.vectors[0].size)
        }
        for ((_, rep) in data.globalReps) {
            if (!rep.isEmpty()) {
                return Int64(rep.size)
            }
        }
    }
    return 0
}

func countTestVectors(dataset: HashMap<String, FolderData>, nFolds: Int64, repMethod: String): Int64 {
    var total: Int64 = 0
    for (fold in 0..nFolds) {
        let splits = buildFoldSplits(dataset, fold, nFolds, repMethod)
        for (sp in splits) {
            total += Int64(sp.testVecs.size)
        }
    }
    return total
}

func countTestGroups(dataset: HashMap<String, FolderData>, nFolds: Int64, repMethod: String): Int64 {
    var total: Int64 = 0
    for (fold in 0..nFolds) {
        let splits = buildFoldSplits(dataset, fold, nFolds, repMethod)
        for (sp in splits) {
            if (!sp.testVecs.isEmpty()) {
                total += 1
            }
        }
    }
    return total
}

func buildVectormapFromDataset(
    dataset: HashMap<String, FolderData>,
    scale: EvalScale
): (VectorMap, Int64, Int64) {
    let vectormap = VectorMap()
    var totalVectors: Int64 = 0
    var dim: Int64 = 0
    for (folder in sortedFolderNames(dataset)) {
        let data = dataset[folder]
        if (data.vectors.isEmpty()) { continue }
        if (!data.globalReps.contains(scale.repMethod)) { continue }
        let repVec = data.globalReps[scale.repMethod]
        let vSize = data.vectors.size
        if (dim == 0 && vSize > 0) {
            dim = Int64(data.vectors[0].size)
        }
        let vecArr = Array<Array<Float64>>(vSize, { i => data.vectors[i] })
        let descArr = Array<Any>(vSize, { i => data.descriptions[i] })
        let group = VectorGroup(
            folder, repVec, scale.repMethod, vecArr, descArr,
            vectorType: scale.vectorType, groupType: "default"
        )
        vectormap.insert(data.key, group, metadataOpt: None)
        totalVectors += Int64(vSize)
    }
    let keyVectors = HashMap<String, Array<Float64>>()
    for (folder in sortedFolderNames(dataset)) {
        let data = dataset[folder]
        if (data.globalReps.contains(scale.repMethod)) {
            keyVectors[data.key] = data.globalReps[scale.repMethod]
        }
    }
    vectormap.setKeyVectors(keyVectors)
    vectormap.buildHdmg(embeddingK: 12, semanticIntraK: 20, semanticBridgeKeys: 2, semanticBridgePerKey: 1)
    return (vectormap, Int64(vectormap.data.size), totalVectors)
}

func benchBaseKey(key: String): String {
    let idxOpt = key.lastIndexOf("-")
    if (idxOpt.isNone()) { return key }
    let idx = idxOpt.getOrThrow()
    if (idx <= 0 || idx >= key.size - 1) { return key }
    let suffix = key[(idx + 1)..key.size]
    try {
        let _ = Int64.parse(suffix)
        return key[0..idx]
    } catch (e: Exception) {
        return key
    }
}

func resultId(result: SearchResult): String {
    if (result.vectorIdx.isNone()) { return "" }
    let idx = result.vectorIdx.getOrThrow()
    if (idx < 0 || idx >= result.group.descriptions.size) { return "" }
    let desc = result.group.descriptions[idx]
    if (desc.contains("img_path")) {
        let imgOpt = desc["img_path"] as String
        if (imgOpt.isSome()) { return imgOpt.getOrThrow() }
    }
    if (desc.contains("text")) {
        let textOpt = desc["text"] as String
        if (textOpt.isSome()) { return textOpt.getOrThrow() }
    }
    if (desc.contains("id")) {
        let idOpt = desc["id"] as String
        if (idOpt.isSome()) { return idOpt.getOrThrow() }
    }
    return ""
}

func resultBaseKeys(results: Array<SearchResult>, topK: Int64): ArrayList<String> {
    let out = ArrayList<String>()
    let limit = if (topK < results.size) { topK } else { results.size }
    for (i in 0..limit) { out.add(benchBaseKey(results[i].key)) }
    return out
}

func flatBaseKeys(results: ArrayList<FlatScore>, topK: Int64): ArrayList<String> {
    let out = ArrayList<String>()
    let limit = if (topK < results.size) { topK } else { results.size }
    for (i in 0..limit) { out.add(benchBaseKey(results[i].key)) }
    return out
}

func recallIdOverlap(gt: ArrayList<FlatScore>, resultIds: ArrayList<String>): Float64 {
    if (gt.isEmpty()) { return 0.0 }
    var hits: Int64 = 0
    for (g in gt) {
        if (resultIds.contains(g.id)) { hits++ }
    }
    return Float64(hits) / Float64(gt.size)
}

func recallKeyHitRate(gtKeys: ArrayList<String>, resultKeys: ArrayList<String>, topK: Int64): Float64 {
    if (topK <= 0) { return 0.0 }
    var hits: Int64 = 0
    for (rk in resultKeys) {
        if (gtKeys.contains(rk)) { hits++ }
    }
    return Float64(hits) / Float64(topK)
}

public func runStorageEvaluation(
    dataset: HashMap<String, FolderData>,
    scale: EvalScale,
    datasetLabel: String
): Unit {
    let (vectormap, totalKeys, totalVectors) = buildVectormapFromDataset(dataset, scale)
    let dim = datasetDimension(dataset)

    println("\n=== [${datasetLabel}] VectorMap Storage Analysis ===")
    println("Total keys: ${totalKeys}")
    let keyList = ArrayList<String>()
    for (folder in sortedFolderNames(dataset)) {
        let data = dataset[folder]
        if (!data.vectors.isEmpty()) {
            keyList.add(data.key)
        }
    }
    println("Keys: ${keyList}")

    println("\nVector groups detail:")
    println("${padRight("Key", 15)} ${padRight("Group", 30)} ${padRight("Vectors", 8)} ${padRight("Dimension", 10)} ${padRight("Type", 10)}")
    println("-" * 80)

    for (folder in sortedFolderNames(dataset)) {
        let data = dataset[folder]
        if (data.vectors.isEmpty()) { continue }
        if (!data.globalReps.contains(scale.repMethod)) { continue }
        let vCount = Int64(data.vectors.size)
        let vDim = if (vCount > 0) { Int64(data.vectors[0].size) } else { dim }
        println("${padRight(data.key, 15)} ${padRight(folder, 30)} ${padRight(vCount.toString(), 8)} ${padRight(vDim.toString(), 10)} ${padRight(scale.vectorType, 10)}")
    }

    let avgPerKey = if (totalKeys > 0) {
        Float64(totalVectors) / Float64(totalKeys)
    } else { 0.0 }
    println("\nSummary:")
    println("  Total keys: ${totalKeys}")
    println("  Total number of vectors: ${totalVectors}")
    println("  Average vectors per key: ${formatFloat4(avgPerKey)}")
    println("  VectorMap groups inserted: ${Int64(vectormap.data.size)}")
}

class FoldSplit {
    FoldSplit(
        let folder: String,
        let key: String,
        let trainVecs: ArrayList<Array<Float64>>,
        let testVecs: ArrayList<Array<Float64>>,
        let trainDescs: ArrayList<Any>,
        let testDescs: ArrayList<Any>,
        let trainRecordIds: ArrayList<Any>,
        let testRecordIds: ArrayList<Any>,
        let trainRep: Array<Float64>,
        let testRep: Array<Float64>
    ) {}
}

func recordIndex(index: Int64): String {
    let raw = index.toString()
    var out = raw
    for (_ in 0..(8 - raw.size)) { out = "0" + out }
    return out
}

func buildFoldSplits(
    dataset: HashMap<String, FolderData>,
    fold: Int64,
    nFolds: Int64,
    repMethod: String
): ArrayList<FoldSplit> {
    let splits = ArrayList<FoldSplit>()
    for (folder in sortedFolderNames(dataset)) {
        let data = dataset[folder]
        let nTotal = data.vectors.size
        if (nTotal == 0) { continue }
        if (!data.foldReps.contains(repMethod)) { continue }
        if (fold >= data.foldReps[repMethod].size) { continue }

        let foldSize = if (nFolds > 0) { nTotal / nFolds } else { 0 }
        let startIdx = fold * foldSize
        let endIdx = if (fold < nFolds - 1) { startIdx + foldSize } else { nTotal }
        let foldRep = data.foldReps[repMethod][fold]

        let trainVecs = ArrayList<Array<Float64>>()
        let testVecs = ArrayList<Array<Float64>>()
        let trainDescs = ArrayList<Any>()
        let testDescs = ArrayList<Any>()
        let trainRecordIds = ArrayList<Any>()
        let testRecordIds = ArrayList<Any>()
        for (i in 0..nTotal) {
            let stableId = "${folder}/${recordIndex(i)}"
            if (i >= startIdx && i < endIdx) {
                testVecs.add(data.vectors[i])
                testDescs.add(data.descriptions[i])
                testRecordIds.add(stableId)
            } else {
                trainVecs.add(data.vectors[i])
                trainDescs.add(data.descriptions[i])
                trainRecordIds.add(stableId)
            }
        }
        if (trainVecs.isEmpty() && testVecs.isEmpty()) { continue }
        splits.add(FoldSplit(folder, data.key, trainVecs, testVecs, trainDescs, testDescs,
            trainRecordIds, testRecordIds,
            foldRep.trainRep, foldRep.testRep))
    }
    return splits
}

func buildVectormapForFold(splits: ArrayList<FoldSplit>, vType: String, repMethod: String): (VectorMap, HashMap<String, ArrayList<VectorGroup>>) {
    let vectormap = VectorMap()
    let testGroups = HashMap<String, ArrayList<VectorGroup>>()
    let keyVectors = HashMap<String, Array<Float64>>()
    for (sp in splits) {
        if (!sp.trainVecs.isEmpty()) {
            let trSize = sp.trainVecs.size
            let vecArr = Array<Array<Float64>>(trSize, { i => sp.trainVecs[i] })
            let descArr = Array<Any>(trSize, { i => sp.trainDescs[i] })
            let group = VectorGroup(sp.folder, sp.trainRep, repMethod, vecArr, descArr,
                vectorType: vType, groupType: "")
            vectormap.insert(sp.key, group, metadataOpt: None)
            keyVectors[sp.key] = sp.trainRep
        }
        if (!sp.testVecs.isEmpty()) {
            let teSize = sp.testVecs.size
            let vecArr = Array<Array<Float64>>(teSize, { i => sp.testVecs[i] })
            let descArr = Array<Any>(teSize, { i => sp.testDescs[i] })
            let testGroup = VectorGroup(sp.folder + "_test", sp.testRep, repMethod, vecArr, descArr,
                vectorType: vType, groupType: "")
            if (!testGroups.contains(sp.key)) {
                testGroups[sp.key] = ArrayList<VectorGroup>()
            }
            testGroups[sp.key].add(testGroup)
        }
    }
    vectormap.setKeyVectors(keyVectors)
    vectormap.buildHdmg(embeddingK: 12, semanticIntraK: 20, semanticBridgeKeys: 2, semanticBridgePerKey: 1)
    return (vectormap, testGroups)
}

func buildAlignedFoldState(
    splits: ArrayList<FoldSplit>,
    vType: String,
    repMethod: String,
    config: AlignedBenchmarkConfig
): (VectorMap, HashMap<String, ArrayList<VectorGroup>>, ArrayList<FlatItem>) {
    let vectormap = VectorMap()
    let testGroups = HashMap<String, ArrayList<VectorGroup>>()
    let keyVectors = HashMap<String, Array<Float64>>()
    let flatItems = ArrayList<FlatItem>()

    for (sp in splits) {
        if (!sp.trainVecs.isEmpty()) {
            let trSize = sp.trainVecs.size
            let vecArr = Array<Array<Float64>>(trSize, { i => sp.trainVecs[i] })
            let descArr = Array<Any>(trSize, { i => sp.trainRecordIds[i] })
            let sourceGroup = VectorGroup(sp.folder, sp.trainRep, repMethod, vecArr, descArr,
                vectorType: vType, groupType: "")
            vectormap.insertWithAutoCluster(sp.key, sourceGroup, metadataOpt: None, alpha: config.clusterAlpha)
            keyVectors[sp.key] = sp.trainRep
            for (i in 0..trSize) {
                let raw = sp.trainRecordIds[i]
                let sOpt = raw as String
                let id = if (sOpt.isSome()) { sOpt.getOrThrow() } else { "${sp.key}#train#${i}" }
                flatItems.add(FlatItem(id, sp.key, sp.trainVecs[i]))
            }
        }
        if (!sp.testVecs.isEmpty()) {
            let teSize = sp.testVecs.size
            let vecArr = Array<Array<Float64>>(teSize, { i => sp.testVecs[i] })
            let descArr = Array<Any>(teSize, { i => sp.testRecordIds[i] })
            let testGroup = VectorGroup(sp.folder + "_test", sp.testRep, repMethod, vecArr, descArr,
                vectorType: vType, groupType: "")
            if (!testGroups.contains(sp.key)) {
                testGroups[sp.key] = ArrayList<VectorGroup>()
            }
            testGroups[sp.key].add(testGroup)
        }
    }
    vectormap.setKeyVectors(keyVectors)
    vectormap.buildHdmgWithConfig(HdmgBuildConfig(
        embeddingK: config.embeddingK,
        semanticIntraK: config.semanticIntraK,
        semanticBridgeKeys: config.semanticBridgeKeys,
        semanticBridgePerKey: config.semanticBridgePerKey,
        useMutualEmbedding: config.useMutualEmbedding
    ))
    return (vectormap, testGroups, flatItems)
}

func flatMixedSearch(
    flatItems: ArrayList<FlatItem>,
    queryVector: Array<Float64>,
    queryKeyVector: Option<Array<Float64>>,
    keyVectors: HashMap<String, Array<Float64>>,
    beta: Float64,
    topK: Int64
): ArrayList<FlatScore> {
    let scores = ArrayList<FlatScore>()
    for (item in flatItems) {
        let embD = cosineDistance(queryVector, item.vector)
        var score = embD
        if (beta != 0.0 && queryKeyVector.isSome()) {
            let base = benchBaseKey(item.key)
            var kvOpt: Option<Array<Float64>> = None
            if (keyVectors.contains(base)) {
                kvOpt = Some(keyVectors[base])
            } else if (keyVectors.contains(item.key)) {
                kvOpt = Some(keyVectors[item.key])
            }
            if (kvOpt.isSome()) {
                let semD = cosineDistance(queryKeyVector.getOrThrow(), kvOpt.getOrThrow())
                score = beta * semD + (1.0 - beta) * embD
            } else {
                score = (1.0 - beta) * embD
            }
        }
        scores.add(FlatScore(score, item.id, item.key))
    }
    sort(scores, by: { a, b => a.compare(b) })
    let limit = if (topK < scores.size) { topK } else { scores.size }
    let out = ArrayList<FlatScore>()
    for (i in 0..limit) { out.add(scores[i]) }
    return out
}

func flatEmbeddingSearch(
    flatItems: ArrayList<FlatItem>,
    queryVector: Array<Float64>,
    topK: Int64
): ArrayList<FlatScore> {
    return flatMixedSearch(flatItems, queryVector, None, HashMap<String, Array<Float64>>(), 0.0, topK)
}

func searchResultIds(results: Array<SearchResult>, topK: Int64): ArrayList<String> {
    let ids = ArrayList<String>()
    let limit = if (topK < results.size) { topK } else { results.size }
    for (i in 0..limit) {
        let id = resultId(results[i])
        if (!id.isEmpty()) { ids.add(id) }
    }
    return ids
}

func parityResultTrace(results: Array<SearchResult>, topK: Int64): String {
    var out = ""
    let limit = if (topK < results.size) { topK } else { results.size }
    for (i in 0..limit) {
        if (i > 0) { out += ";" }
        let result = results[i]
        let id = resultId(result)
        out += "${benchBaseKey(result.key)}::${id}"
    }
    return out
}

func parityFlatTrace(results: ArrayList<FlatScore>, topK: Int64): String {
    var out = ""
    let limit = if (topK < results.size) { topK } else { results.size }
    for (i in 0..limit) {
        if (i > 0) { out += ";" }
        out += "${benchBaseKey(results[i].key)}::${results[i].id}"
    }
    return out
}

func recordHdmgTrace(vectormap: VectorMap, trace: HdmgTraceStats): Unit {
    let raw = vectormap.getLastHdmgSearchStats()
    if (raw.isEmpty()) { return }
    let modeOpt = raw["mode"] as String
    if (modeOpt.isNone() || modeOpt.getOrThrow() != "hdmg") { return }

    var depthSum = 0.0
    var depthCount: Int64 = 0
    let depthsOpt = raw["route_depths"] as ArrayList<Int64>
    if (depthsOpt.isSome()) {
        let depths = depthsOpt.getOrThrow()
        for (d in depths) {
            depthSum += Float64(d)
            depthCount++
        }
    }

    var hopsSum = 0.0
    var hopsCount: Int64 = 0
    let hopsOpt = raw["route_hops"] as ArrayList<Int64>
    if (hopsOpt.isSome()) {
        let hops = hopsOpt.getOrThrow()
        for (h in hops) {
            hopsSum += Float64(h)
            hopsCount++
        }
    }

    trace.queries++
    if (depthCount > 0) {
        trace.totalDepth += depthSum / Float64(depthCount)
    }
    if (hopsCount > 0) {
        trace.totalHops += hopsSum / Float64(hopsCount)
    }

    let candOpt = raw["candidate_pool_size"] as Int64
    if (candOpt.isSome()) { trace.totalCandidatePool += Float64(candOpt.getOrThrow()) }
    let selectedOpt = raw["selected_nodes"] as Int64
    if (selectedOpt.isSome()) { trace.totalSelectedNodes += Float64(selectedOpt.getOrThrow()) }
    let callsOpt = raw["mixed_score_calls"] as Int64
    if (callsOpt.isSome()) { trace.totalMixedScoreCalls += Float64(callsOpt.getOrThrow()) }
    let entryNsOpt = raw["t_entry_ns"] as Int64
    if (entryNsOpt.isSome()) { trace.totalEntryNs += entryNsOpt.getOrThrow() }
    let walkNsOpt = raw["t_walk_ns"] as Int64
    if (walkNsOpt.isSome()) { trace.totalWalkNs += walkNsOpt.getOrThrow() }
    let rerankNsOpt = raw["t_rerank_ns"] as Int64
    if (rerankNsOpt.isSome()) { trace.totalRerankNs += rerankNsOpt.getOrThrow() }
    let totalNsOpt = raw["t_total_ns"] as Int64
    if (totalNsOpt.isSome()) { trace.totalSearchNs += totalNsOpt.getOrThrow() }
}

func evalModeSingleQuery(
    dataset: HashMap<String, FolderData>,
    nFolds: Int64,
    vType: String,
    repMethod: String,
    mode: String,
    topK: Int64,
    maxQueries: Int64,
    stats: MethodRecallStats,
    progressLabel: String
): Unit {
    var done: Int64 = 0
    let progressEvery: Int64 = 20
    for (fold in 0..nFolds) {
        if (maxQueries > 0 && done >= maxQueries) { break }
        let splits = buildFoldSplits(dataset, fold, nFolds, repMethod)
        let (vectormap, _) = buildVectormapForFold(splits, vType, repMethod)
        for (sp in splits) {
            for (vec in sp.testVecs) {
                if (maxQueries > 0 && done >= maxQueries) { break }
                let results = vectormap.search(vec, keyOpt: None, topK: topK, mode: mode,
                    distanceMethod: "cosine", vectorType: None, groupType: None, groupName: None)
                stats.recalls.add(recallKeyAtK(results, sp.key, topK))
                stats.ndcgs.add(keyNdcgAtK(results, sp.key, topK))
                done++
                if (maxQueries > 0 && done % progressEvery == 0) {
                    println("Under test: ${done}/${maxQueries} (${progressLabel})")
                }
            }
        }
    }
    if (done > 0 && (maxQueries <= 0 || done % progressEvery != 0)) {
        let cap = if (maxQueries > 0) { maxQueries } else { done }
        println("Under test: ${done}/${cap} (${progressLabel})")
    }
}

func evalModeGroupN(
    dataset: HashMap<String, FolderData>,
    nFolds: Int64,
    vType: String,
    repMethod: String,
    distMethod: String,
    topK: Int64,
    maxQueries: Int64,
    stats: MethodRecallStats,
    progressLabel: String
): Unit {
    var done: Int64 = 0
    let progressEvery: Int64 = 20
    for (fold in 0..nFolds) {
        if (maxQueries > 0 && done >= maxQueries) { break }
        let splits = buildFoldSplits(dataset, fold, nFolds, repMethod)
        let (vectormap, _) = buildVectormapForFold(splits, vType, repMethod)
        for (sp in splits) {
            if (maxQueries > 0 && done >= maxQueries) { break }
            if (sp.testVecs.isEmpty()) { continue }
            let qVecs = sp.testVecs.toArray()
            let results = vectormap.searchMulti(qVecs, keyOpt: None, topK: topK,
                groupNDistanceMethod: distMethod, distanceMethod: "cosine",
                vectorType: None, groupType: None, groupName: None)
            stats.recalls.add(recallKeyAtK(results, sp.key, topK))
            stats.ndcgs.add(keyNdcgAtK(results, sp.key, topK))
            done++
            if (maxQueries > 0 && done % progressEvery == 0) {
                println("Under test: ${done}/${maxQueries} (${progressLabel})")
            }
        }
    }
    if (done > 0 && (maxQueries <= 0 || done % progressEvery != 0)) {
        let cap = if (maxQueries > 0) { maxQueries } else { done }
        println("Under test: ${done}/${cap} (${progressLabel})")
    }
}

func evalModeHdmg(
    dataset: HashMap<String, FolderData>,
    nFolds: Int64,
    vType: String,
    repMethod: String,
    topK: Int64,
    maxQueries: Int64,
    stats: MethodRecallStats,
    trace: HdmgTraceStats,
    progressLabel: String
): Unit {
    var done: Int64 = 0
    let progressEvery: Int64 = 20
    for (fold in 0..nFolds) {
        if (maxQueries > 0 && done >= maxQueries) { break }
        let splits = buildFoldSplits(dataset, fold, nFolds, repMethod)
        let (vectormap, testGroups) = buildVectormapForFold(splits, vType, repMethod)
        for (sp in splits) {
            if (maxQueries > 0 && done >= maxQueries) { break }
            if (!testGroups.contains(sp.key)) { continue }
            let groups = testGroups[sp.key]
            for (group in groups) {
                for (vec in group.vectors) {
                    if (maxQueries > 0 && done >= maxQueries) { break }
                    let results = vectormap.searchHdmg(
                        vec,
                        queryKeyVector: Some(sp.testRep),
                        alpha: 0.5,
                        topK: topK,
                        maxSteps: 100,
                        distanceMethod: "cosine",
                        clusterPoolSize: Some(topK * 3),
                        topKeyCandidates: 5,
                        extraHops: 0
                    )
                    recordHdmgTrace(vectormap, trace)
                    stats.recalls.add(recallKeyAtK(results, sp.key, topK))
                    stats.ndcgs.add(keyNdcgAtK(results, sp.key, topK))
                    done++
                    if (maxQueries > 0 && done % progressEvery == 0) {
                        println("Under test: ${done}/${maxQueries} (${progressLabel})")
                    }
                }
            }
        }
    }
    if (done > 0 && (maxQueries <= 0 || done % progressEvery != 0)) {
        let cap = if (maxQueries > 0) { maxQueries } else { done }
        println("Under test: ${done}/${cap} (${progressLabel})")
    }
}

func printRecallResultsTable(methodStats: ArrayList<MethodRecallStats>, topK: Int64): Unit {
    var totalQueries: Int64 = 0
    for (st in methodStats) {
        if (Int64(st.recalls.size) > totalQueries) { totalQueries = Int64(st.recalls.size) }
    }

    println("\n" + "=" * 92)
    println("=== Recall/NDCG Test results ===")
    println("=" * 92)
    println("Total queries: ${totalQueries}")
    println("Number of successful queries: ${totalQueries}")
    println("| ${padRight("Method", 22)} | ${padRight("Avg Recall@${topK}", 17)} | ${padRight("NDCG@${topK}", 10)} | ${padRight("Min Recall", 10)} | ${padRight("Max Recall", 10)} |")
    println("|-----------------------|-------------------|------------|------------|-----------|")
    for (st in methodStats) {
        if (st.recalls.isEmpty()) {
            println("| ${padRight(st.label, 22)} | N/A               | N/A        | N/A        | N/A        |")
        } else {
            println("| ${padRight(st.label, 22)} | ${formatFloat4(st.avg())}         | ${formatFloat4(st.avgNdcg())}     | ${formatFloat4(st.minVal())}   | ${formatFloat4(st.maxVal())}   |")
        }
    }
    println("=" * 92)
}

func printHdmgTraceStats(trace: HdmgTraceStats): Unit {
    if (trace.queries == 0) { return }
    println("\n--- HDMG graph trace statistics ---")
    println("HDMG (steps): ${formatFloat2(trace.avgDepth())}")
    println("HDMG (hops): ${formatFloat2(trace.avgHops())}")
    println("HDMG Average candidate pool size: ${formatFloat2(trace.avgCandidatePool())}")
    println("HDMG Average final number of micro-clusters: ${formatFloat2(trace.avgSelectedNodes())}")
    println("HDMG Average number of mixed_score calls: ${formatFloat2(trace.avgMixedScoreCalls())}")
}

public func runSearchEvaluation(
    dataset: HashMap<String, FolderData>,
    scale: EvalScale,
    datasetLabel: String
): Unit {
    let nFolds = scale.nFolds
    let maxQ = scale.maxQueries
    let vType = scale.vectorType
    let repMethod = scale.repMethod
    let topK = if (scale.topK > 0) { scale.topK } else { defaultTopK }

    let totalTestVecs = countTestVectors(dataset, nFolds, repMethod)
    let totalTestGroups = countTestGroups(dataset, nFolds, repMethod)

    println("\n" + "=" * 80)
    println("=== [${datasetLabel}] Multiple sets of parameters Recall test ===")
    println("Testing 1 evaluation suite (${nFolds}-fold cross-validation, Cangjie VectorMap)")
    println("rep_method: ${repMethod}")
    println("top_k: ${topK}")
    println("=" * 80)

    println("\n--- No. 1/1 Group parameters ---")
    println("eval: cross-validation")

    println("\n" + "=" * 80)
    println("=== Recall test ===")
    println("Ground Truth: same-class key match (hold-out fold, top ${topK})")
    println("single: VectorMap single-vector search top ${topK}")
    println("group1: VectorMap group1 search top ${topK}")
    println("Representative: representative vector search top ${topK}")
    println("groupN (hausdorff/maxsim): multi-vector group search top ${topK}")
    println("HDMG: mixed key+representative search top ${topK}")
    println("=" * 80)

    println("\n=== Parameter settings ===")
    println("TOP_K=${topK}")
    println("NFOLDS=${nFolds}")
    println("REP_METHOD=${repMethod}")
    println("VECTOR_TYPE=${vType}")
    println("MAX_QUERIES=${if (maxQ > 0) { maxQ.toString() } else { "all" }}")
    println("=" * 80)

    println("\ntotal ${totalTestVecs} test vectors (${totalTestGroups} test groups for groupN)")
    if (maxQ > 0) {
        println("Only before testing ${maxQ} per mode (max_queries=${maxQ})\n")
    } else {
        println("test all queries per mode\n")
    }

    let methodStats = ArrayList<MethodRecallStats>()

    let stSingle = MethodRecallStats("single")
    evalModeSingleQuery(dataset, nFolds, vType, repMethod, "single", topK, maxQ, stSingle, "single")
    methodStats.add(stSingle)

    let stGroup1 = MethodRecallStats("group1")
    evalModeSingleQuery(dataset, nFolds, vType, repMethod, "group1", topK, maxQ, stGroup1, "group1")
    methodStats.add(stGroup1)

    let stRep = MethodRecallStats("Representative")
    evalModeSingleQuery(dataset, nFolds, vType, repMethod, "representative", topK, maxQ, stRep, "representative")
    methodStats.add(stRep)

    let stHaus = MethodRecallStats("groupN (hausdorff)")
    let maxGroupN = if (maxQ > 0) { maxQ / 2 } else { 0 }
    evalModeGroupN(dataset, nFolds, vType, repMethod, "hausdorff", topK, maxGroupN, stHaus, "hausdorff")
    methodStats.add(stHaus)

    let stMax = MethodRecallStats("groupN (maxsim)")
    evalModeGroupN(dataset, nFolds, vType, repMethod, "maxsim", topK, maxGroupN, stMax, "maxsim")
    methodStats.add(stMax)

    let stHdmg = MethodRecallStats("HDMG")
    let hdmgTrace = HdmgTraceStats()
    evalModeHdmg(dataset, nFolds, vType, repMethod, topK, maxQ, stHdmg, hdmgTrace, "hdmg")
    methodStats.add(stHdmg)

    printRecallResultsTable(methodStats, topK)
    printHdmgTraceStats(hdmgTrace)
}

func evalAlignedBeta(
    dataset: HashMap<String, FolderData>,
    scale: EvalScale,
    beta: Float64,
    config: AlignedBenchmarkConfig
): BetaEvalStats {
    let stats = BetaEvalStats(beta)
    let topK = if (scale.topK > 0) { scale.topK } else { defaultTopK }
    let hdmgSearchConfig = HdmgSearchConfig(
        maxSteps: config.maxSteps,
        entryAlphaThreshold: config.entryAlphaThreshold,
        clusterPoolMultiplier: config.clusterPoolMultiplier,
        topKeyCandidates: config.topKeyCandidates,
        extraHops: config.extraHops
    )
    var done: Int64 = 0
    let useKeyRecall = beta > 0.999999
    for (fold in 0..scale.nFolds) {
        if (scale.maxQueries > 0 && done >= scale.maxQueries) { break }
        let splits = buildFoldSplits(dataset, fold, scale.nFolds, scale.repMethod)
        let buildStartNs = nowNs()
        let (vectormap, testGroups, flatItems) = buildAlignedFoldState(splits, scale.vectorType, scale.repMethod, config)
        stats.totalBuildNs += nowNs() - buildStartNs
        stats.buildCount++
        for (sp in splits) {
            if (scale.maxQueries > 0 && done >= scale.maxQueries) { break }
            if (!testGroups.contains(sp.key)) { continue }
            let groups = testGroups[sp.key]
            for (group in groups) {
                for (vec in group.vectors) {
                    if (scale.maxQueries > 0 && done >= scale.maxQueries) { break }
                    let gt = flatMixedSearch(flatItems, vec, Some(sp.testRep), vectormap.keyVectors, beta, topK)

                    let mixedStartNs = nowNs()
                    let mixedResults = vectormap.searchHdmgWithConfig(
                        vec,
                        Some(sp.testRep),
                        beta,
                        topK,
                        hdmgSearchConfig,
                        distanceMethod: "cosine"
                    )
                    let mixedEndNs = nowNs()
                    stats.mixed.addLatency(mixedEndNs - mixedStartNs)
                    recordHdmgTrace(vectormap, stats.hdmgTrace)

                    let repStartNs = nowNs()
                    let repResults = vectormap.searchWithRepresentativeRerank(
                        vec,
                        queryKeyVector: Some(sp.testRep),
                        beta: beta,
                        topK: topK,
                        numGroups: Some(topK),
                        distanceMethod: "cosine"
                    )
                    let repEndNs = nowNs()
                    stats.representative.addLatency(repEndNs - repStartNs)

                    let pyflatStartNs = nowNs()
                    let pyflatResults = flatEmbeddingSearch(flatItems, vec, topK)
                    let pyflatEndNs = nowNs()
                    stats.pyflat.addLatency(pyflatEndNs - pyflatStartNs)

                    if (useKeyRecall) {
                        let gtKeys = flatBaseKeys(gt, topK)
                        let mixedKeys = resultBaseKeys(mixedResults, topK)
                        let repKeys = resultBaseKeys(repResults, topK)
                        let flatKeys = flatBaseKeys(pyflatResults, topK)
                        stats.mixed.recalls.add(recallKeyHitRate(gtKeys, mixedKeys, topK))
                        stats.representative.recalls.add(recallKeyHitRate(gtKeys, repKeys, topK))
                        stats.pyflat.recalls.add(recallKeyHitRate(gtKeys, flatKeys, topK))
                        stats.mixed.ndcgs.add(binaryNdcgAtK(gtKeys, mixedKeys, topK))
                        stats.representative.ndcgs.add(binaryNdcgAtK(gtKeys, repKeys, topK))
                        stats.pyflat.ndcgs.add(binaryNdcgAtK(gtKeys, flatKeys, topK))
                    } else {
                        let gtIds = ArrayList<String>()
                        for (r in gt) { gtIds.add(r.id) }
                        let mixedIds = searchResultIds(mixedResults, topK)
                        let repIds = searchResultIds(repResults, topK)
                        stats.mixed.recalls.add(recallIdOverlap(gt, mixedIds))
                        stats.representative.recalls.add(recallIdOverlap(gt, repIds))
                        let pyIds = ArrayList<String>()
                        for (r in pyflatResults) { pyIds.add(r.id) }
                        stats.pyflat.recalls.add(recallIdOverlap(gt, pyIds))
                        stats.mixed.ndcgs.add(binaryNdcgAtK(gtIds, mixedIds, topK))
                        stats.representative.ndcgs.add(binaryNdcgAtK(gtIds, repIds, topK))
                        stats.pyflat.ndcgs.add(binaryNdcgAtK(gtIds, pyIds, topK))
                    }

                    done++
                    if (scale.maxQueries > 0 && done % 20 == 0) {
                        println("Under test: ${done}/${scale.maxQueries} (beta=${formatBeta(beta)})")
                    }
                }
            }
        }
    }
    return stats
}

func printAlignedBetaDetail(stats: BetaEvalStats, topK: Int64): Unit {
    println("\n" + "=" * 92)
    println("=== Recall/NDCG Test results ===")
    println("=" * 92)
    println("Total queries: ${stats.mixed.recalls.size}")
    println("Number of successful queries: ${stats.mixed.recalls.size}")
    println("| ${padRight("Method", 22)} | ${padRight("Avg Recall@${topK}", 17)} | ${padRight("NDCG@${topK}", 10)} | ${padRight("Min Recall", 10)} | ${padRight("Max Recall", 10)} |")
    println("|-----------------------|-------------------|------------|------------|-----------|")
    println("| ${padRight("Mixed (beta=${formatBeta(stats.beta)}, HDMG)", 22)} | ${formatFloat4(stats.mixed.avg())}         | ${formatFloat4(stats.mixed.avgNdcg())}     | ${formatFloat4(stats.mixed.minVal())}   | ${formatFloat4(stats.mixed.maxVal())}   |")
    println("| ${padRight("Representative", 22)} | ${formatFloat4(stats.representative.avg())}         | ${formatFloat4(stats.representative.avgNdcg())}     | ${formatFloat4(stats.representative.minVal())}   | ${formatFloat4(stats.representative.maxVal())}   |")
    println("| ${padRight("ExactSearch", 22)} | ${formatFloat4(stats.pyflat.avg())}         | ${formatFloat4(stats.pyflat.avgNdcg())}     | ${formatFloat4(stats.pyflat.minVal())}   | ${formatFloat4(stats.pyflat.maxVal())}   |")
    println("=" * 92)
    printHdmgTraceStats(stats.hdmgTrace)
}

func printAlignedSummary(results: ArrayList<BetaEvalStats>, topK: Int64): Unit {
    println("\n" + "=" * 120)
    println("=== (alpha=0.5, Recall@${topK}) ===")
    println("=" * 120)
    println("${padRight("exp_factor", 12)} ${padRight("beta", 6)} | ${padRight("mixed_recall", 13)} ${padRight("min", 6)} ${padRight("max", 6)} | ${padRight("representative", 12)} ${padRight("min", 6)} ${padRight("max", 6)} | ${padRight("pyflat", 8)} ${padRight("milvus", 8)} ${padRight("milvus_mix", 10)} ${padRight("qdrant", 8)} ${padRight("qdrant_mix", 10)} ${padRight("chroma", 8)} ${padRight("chroma_mix", 12)}")
    println("-" * 140)
    for (r in results) {
        println("${padRight("3", 12)} ${padRight(formatBeta(r.beta), 6)} | ${padRight(formatFloat4(r.mixed.avg()), 13)} ${padRight(formatFloat2(r.mixed.minVal()), 6)} ${padRight(formatFloat2(r.mixed.maxVal()), 6)} | ${padRight(formatFloat4(r.representative.avg()), 12)} ${padRight(formatFloat2(r.representative.minVal()), 6)} ${padRight(formatFloat2(r.representative.maxVal()), 6)} | ${padRight(formatFloat4(r.pyflat.avg()), 8)} ${padRight("N/A", 8)} ${padRight("N/A", 10)} ${padRight("N/A", 8)} ${padRight("N/A", 10)} ${padRight("N/A", 8)} ${padRight("N/A", 12)}")
    }
    println("\n" + "=" * 140)
    println("=== (latency: ms/query) ===")
    println("=" * 140)
    println("${padRight("exp_factor", 12)} ${padRight("beta", 6)} | ${padRight("hdmg", 12)} ${padRight("representative", 14)} ${padRight("pyflat", 12)} ${padRight("milvus", 12)} ${padRight("milvus_mix", 12)} ${padRight("qdrant", 12)} ${padRight("qdrant_mix", 12)} ${padRight("chroma", 12)} ${padRight("chroma_mix", 12)}")
    println("-" * 140)
    for (r in results) {
        println("${padRight("3", 12)} ${padRight(formatBeta(r.beta), 6)} | ${padRight(formatMs(r.mixed.avgLatencyMs()), 12)} ${padRight(formatMs(r.representative.avgLatencyMs()), 14)} ${padRight(formatMs(r.pyflat.avgLatencyMs()), 12)} ${padRight("N/A", 12)} ${padRight("N/A", 12)} ${padRight("N/A", 12)} ${padRight("N/A", 12)} ${padRight("N/A", 12)} ${padRight("N/A", 12)}")
    }
    println("=" * 140)

    println("\n=== latency percentiles (ms/query) ===")
    println("${padRight("beta", 6)} | ${padRight("hdmg_p50", 10)} ${padRight("hdmg_p95", 10)} | ${padRight("rep_p50", 10)} ${padRight("rep_p95", 10)} | ${padRight("flat_p50", 10)} ${padRight("flat_p95", 10)}")
    println("-" * 86)
    for (r in results) {
        println("${padRight(formatBeta(r.beta), 6)} | ${padRight(formatMs(r.mixed.percentileLatencyMs(0.50)), 10)} ${padRight(formatMs(r.mixed.percentileLatencyMs(0.95)), 10)} | ${padRight(formatMs(r.representative.percentileLatencyMs(0.50)), 10)} ${padRight(formatMs(r.representative.percentileLatencyMs(0.95)), 10)} | ${padRight(formatMs(r.pyflat.percentileLatencyMs(0.50)), 10)} ${padRight(formatMs(r.pyflat.percentileLatencyMs(0.95)), 10)}")
    }

    println("\n=== index build latency (ms/fold) ===")
    for (r in results) {
        println("beta=${formatBeta(r.beta)} build_pseudo_mean=${formatMs(r.avgBuildMs())} folds=${r.buildCount}")
    }

    println("\n" + "=" * 120)
    println("=== HDMG ( beta) ===")
    println("=" * 120)
    println("${padRight("beta", 6)} | ${padRight("avg_hops", 10)} ${padRight("avg_depth", 10)} ${padRight("cand_pool", 10)} ${padRight("ms_calls", 10)} | ${padRight("t_entry", 10)} ${padRight("t_walk", 10)} ${padRight("t_rerank", 10)} ${padRight("t_total", 10)}")
    println("-" * 120)
    for (r in results) {
        println("${padRight(formatBeta(r.beta), 6)} | ${padRight(formatFloat2(r.hdmgTrace.avgHops()), 10)} ${padRight(formatFloat2(r.hdmgTrace.avgDepth()), 10)} ${padRight(formatFloat2(r.hdmgTrace.avgCandidatePool()), 10)} ${padRight(formatFloat2(r.hdmgTrace.avgMixedScoreCalls()), 10)} | ${padRight(formatMs(r.hdmgTrace.avgEntryMs()), 10)} ${padRight(formatMs(r.hdmgTrace.avgWalkMs()), 10)} ${padRight(formatMs(r.hdmgTrace.avgRerankMs()), 10)} ${padRight(formatMs(r.hdmgTrace.avgSearchMs()), 10)}")
    }
    println("=" * 120)
}

func methodStatsJson(stats: MethodRecallStats): String {
    return "{\"recallAtK\":${formatFloat4(stats.avg())},\"ndcgAtK\":${formatFloat4(stats.avgNdcg())}," +
        "\"latencyMs\":{\"mean\":${formatMs(stats.avgLatencyMs())}," +
        "\"p50\":${formatMs(stats.percentileLatencyMs(0.50))},\"p95\":${formatMs(stats.percentileLatencyMs(0.95))}}}"
}

func alignedResultJson(
    datasetLabel: String,
    scale: EvalScale,
    config: AlignedBenchmarkConfig,
    results: ArrayList<BetaEvalStats>
): String {
    var rows = "["
    for (i in 0..results.size) {
        if (i > 0) { rows += "," }
        let r = results[i]
        rows += "{\"beta\":${formatBeta(r.beta)},\"queryCount\":${r.mixed.recalls.size}," +
            "\"build\":{\"folds\":${r.buildCount},\"meanMs\":${formatMs(r.avgBuildMs())}}," +
            "\"mixed\":${methodStatsJson(r.mixed)}," +
            "\"representative\":${methodStatsJson(r.representative)}," +
            "\"exactSearch\":${methodStatsJson(r.pyflat)}," +
            "\"hdmgTrace\":{\"avgHops\":${formatFloat4(r.hdmgTrace.avgHops())}," +
            "\"avgDepth\":${formatFloat4(r.hdmgTrace.avgDepth())}," +
            "\"avgCandidatePool\":${formatFloat4(r.hdmgTrace.avgCandidatePool())}," +
            "\"avgMixedScoreCalls\":${formatFloat4(r.hdmgTrace.avgMixedScoreCalls())}}}"
    }
    rows += "]"
    return "{\"schemaVersion\":1,\"backend\":\"cangjie-native\",\"dataset\":\"${datasetLabel}\"," +
        "\"scale\":{\"folds\":${scale.nFolds},\"maxFolders\":${scale.maxFolders}," +
        "\"maxQueries\":${scale.maxQueries},\"maxVectorsPerFolder\":${scale.maxVectorsPerFolder}," +
        "\"topK\":${scale.topK},\"vectorType\":\"${scale.vectorType}\",\"repMethod\":\"${scale.repMethod}\"}," +
        "\"config\":{\"clusterAlpha\":${formatBeta(config.clusterAlpha)},\"embeddingK\":${config.embeddingK}," +
        "\"semanticIntraK\":${config.semanticIntraK},\"semanticBridgeKeys\":${config.semanticBridgeKeys}," +
        "\"semanticBridgePerKey\":${config.semanticBridgePerKey},\"useMutualEmbedding\":${config.useMutualEmbedding}," +
        "\"clusterPoolMultiplier\":${config.clusterPoolMultiplier},\"maxSteps\":${config.maxSteps}," +
        "\"entryAlphaThreshold\":${formatBeta(config.entryAlphaThreshold)}," +
        "\"topKeyCandidates\":${config.topKeyCandidates},\"extraHops\":${config.extraHops}},\"results\":${rows}}"
}

public func runAlignedBenchmarkEvaluation(
    dataset: HashMap<String, FolderData>,
    scale: EvalScale,
    datasetLabel: String
): Unit {
    let topK = if (scale.topK > 0) { scale.topK } else { defaultTopK }
    let config = AlignedBenchmarkConfig()
    let betaValues = config.betaValues

    println("\n" + "=" * 80)
    println("=== [${datasetLabel}] Multiple sets of parameters Recall test ===")
    println("Testing ${betaValues.size} parameter combinations")
    println("alpha: ${formatBeta(config.clusterAlpha)}")
    println("hdmg_cluster_pool_size: ${topK * config.clusterPoolMultiplier}")
    println("=" * 80)

    let resultsSummary = ArrayList<BetaEvalStats>()
    for (i in 0..betaValues.size) {
        let beta = betaValues[i]
        println("\n--- No. ${i + 1}/${betaValues.size} Group parameters ---")
        println("beta: ${formatBeta(beta)}")
        println("\n" + "=" * 80)
        println("=== Recall test ===")
        println("Ground Truth: brute-force mixed (beta=${formatBeta(beta)}) Search top ${topK}")
        println("Mixed (beta=${formatBeta(beta)}, HDMG): Mixed key+rep vector search top ${topK}")
        println("Representative: representative rerank top ${topK}")
        println("ExactSearch: full-scan single-vector cosine search top ${topK}")
        println("=" * 80)
        let st = evalAlignedBeta(dataset, scale, beta, config)
        printAlignedBetaDetail(st, topK)
        resultsSummary.add(st)
    }
    printAlignedSummary(resultsSummary, topK)
    println("RESULT_JSON|${alignedResultJson(datasetLabel, scale, config, resultsSummary)}")
}

// 在真实 precomputed 样例上输出逐 query 的机器可读结果，供冻结 Python 参考实现逐项比较。
public func runParityTrace(
    txtPath: String,
    datasetId: String,
    vectorType: String,
    maxQueries: Int64,
    beta: Float64
): Unit {
    if (maxQueries <= 0) { throw IllegalArgumentException("parity maxQueries 必须大于 0") }
    let topK: Int64 = 3
    let config = AlignedBenchmarkConfig()
    let hdmgSearchConfig = HdmgSearchConfig(
        maxSteps: config.maxSteps, entryAlphaThreshold: config.entryAlphaThreshold,
        clusterPoolMultiplier: config.clusterPoolMultiplier,
        topKeyCandidates: config.topKeyCandidates, extraHops: config.extraHops
    )
    let dataset = loadPrecomputedDataset(txtPath, 20)
    let splits = buildFoldSplits(dataset, 0, 5, "simple")
    let (vectormap, testGroups, flatItems) = buildAlignedFoldState(splits, vectorType, "simple", config)
    var done: Int64 = 0
    for (sp in splits) {
        if (done >= maxQueries) { break }
        if (!testGroups.contains(sp.key)) { continue }
        for (group in testGroups[sp.key]) {
            for (queryIndex in 0..group.vectors.size) {
                if (done >= maxQueries) { break }
                let query = group.vectors[queryIndex]
                let queryRaw = group.descriptions[queryIndex]["text"] as String
                let queryDescription = if (queryRaw.isSome()) { queryRaw.getOrThrow() } else { "${queryIndex}" }
                let queryId = "${datasetId}/fold-0/${queryDescription}"
                let exact = flatEmbeddingSearch(flatItems, query, topK)
                let representative = vectormap.searchWithRepresentativeRerank(
                    query, queryKeyVector: Some(sp.testRep), beta: beta, topK: topK,
                    numGroups: Some(topK), distanceMethod: "cosine"
                )
                let mixed = vectormap.searchWithMixedKeyRepVec(
                    query, queryKeyVector: Some(sp.testRep), beta: beta, topK: topK,
                    groupExpansionFactor: config.clusterPoolMultiplier, distanceMethod: "cosine"
                )
                let hdmg = vectormap.searchHdmgWithConfig(
                    query, Some(sp.testRep), beta, topK, hdmgSearchConfig, distanceMethod: "cosine"
                )
                println("PARITY|${queryId}|exact|${parityFlatTrace(exact, topK)}")
                println("PARITY|${queryId}|representative|${parityResultTrace(representative, topK)}")
                println("PARITY|${queryId}|mixed|${parityResultTrace(mixed, topK)}")
                println("PARITY|${queryId}|hdmg|${parityResultTrace(hdmg, topK)}")
                done++
            }
        }
    }
    println("PARITY_SUMMARY|${datasetId}|queries=${done}|beta=${formatBeta(beta)}|topK=${topK}")
}

public func runFullBenchmark(
    txtPath: String,
    datasetLabel: String,
    scale: EvalScale
): Unit {
    println("\n=== Start building VectorMap ===")
    var dataset = loadPrecomputedDataset(txtPath, scale.maxVectorsPerFolder)
    dataset = limitDataset(dataset, scale.maxFolders)

    let dim = datasetDimension(dataset)
    let topK = if (scale.topK > 0) { scale.topK } else { defaultTopK }
    println("[Cache] Precomputed TXT loaded, number of class keys: ${dataset.size}")
    println("precoded ${dataset.size} key (: ${dim})")

    println("\n=== Experimental parameters ===")
    println("TOP_K=${topK}, ALPHA=0.5, BETA=0.3, EXPANSION_FACTOR=3")
    println("HDMG: embedding_k=12, semantic_intra_k=20, bridge_keys=2, bridge_per_key=1, cluster_pool_size=${topK * 3}, extra_hops=0")
    println("PYTHON_FLAT_MODE=naive")
    println("NFOLDS=${scale.nFolds}, REP_METHOD=${scale.repMethod}, VECTOR_TYPE=${scale.vectorType}")
    println("ExactSearch=full-scan single-vector cosine search over all training vectors")
    if (scale.maxQueries > 0) {
        println("MAX_QUERIES=${scale.maxQueries} (per retrieval mode)")
    } else {
        println("MAX_QUERIES=all")
    }
    if (scale.maxVectorsPerFolder > 0) {
        println("MAX_VECTORS_PER_FOLDER=${scale.maxVectorsPerFolder}")
    }

    println("\n=== Test set statistics ===")
    runStorageEvaluation(dataset, scale, datasetLabel)

    let nFolds = scale.nFolds
    let repMethod = scale.repMethod
    let totalTestVecs = countTestVectors(dataset, nFolds, repMethod)
    println("Number of test set folders (classes): ${dataset.size}")
    println("Total number of vectors in test set (all folds): ${totalTestVecs}")

    runAlignedBenchmarkEvaluation(dataset, scale, datasetLabel)
}
