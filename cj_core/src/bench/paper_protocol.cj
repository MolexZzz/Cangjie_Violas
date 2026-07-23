package semanticvg_cj.bench

import std.collection.*
import std.convert.*
import std.core.*
import std.fs.*
import std.math.*
import std.time.*
import semanticvg_cj.storage.*

class PaperTrainRecord {
    let id: String
    let folder: String
    let key: String
    let vector: Array<Float64>
    let clusterKey: String
    let clusterGroup: String
    init(id: String, folder: String, key: String, vector: Array<Float64>,
         clusterKey: String, clusterGroup: String) {
        this.id = id; this.folder = folder; this.key = key; this.vector = vector
        this.clusterKey = clusterKey; this.clusterGroup = clusterGroup
    }
}

class PaperQuery {
    let id: String
    let sourceId: String
    let trueKey: String
    let vector: Array<Float64>
    let keyVector: Array<Float64>
    init(id: String, sourceId: String, trueKey: String, vector: Array<Float64>, keyVector: Array<Float64>) {
        this.id = id; this.sourceId = sourceId; this.trueKey = trueKey
        this.vector = vector; this.keyVector = keyVector
    }
}

class PaperGroundTruth {
    let queryId: String
    let beta: Float64
    let ids: ArrayList<String>
    init(queryId: String, beta: Float64, ids: ArrayList<String>) {
        this.queryId = queryId; this.beta = beta; this.ids = ids
    }
}

class PaperArtifact {
    var protocol: String = ""
    var dataset: String = ""
    var topK: Int64 = 0
    var queryPoolSize: Int64 = 0
    let keys = HashMap<String, Array<Float64>>()
    let records = ArrayList<PaperTrainRecord>()
    let queries = ArrayList<PaperQuery>()
    let groundTruth = ArrayList<PaperGroundTruth>()
}

func paperVector(raw: String): Array<Float64> {
    let parts = raw.split(",")
    return normalizedVector(Array<Float64>(parts.size, { i => Float64.parse(parts[i]) }))
}

func paperIds(raw: String): ArrayList<String> {
    let output = ArrayList<String>()
    if (!raw.isEmpty()) { for (item in raw.split(",")) { output.add(item) } }
    return output
}

func parsePaperLine(raw: String, artifact: PaperArtifact): Unit {
    let line = raw.replace("\r", "")
    if (line.isEmpty()) { return }
    let parts = line.split("\t")
    if (parts[0] == "META") {
        if (parts.size != 4 && parts.size != 5) { throw IllegalArgumentException("invalid META line") }
        artifact.protocol = parts[1]; artifact.dataset = parts[2]; artifact.topK = Int64.parse(parts[3])
        if (parts.size == 5) { artifact.queryPoolSize = Int64.parse(parts[4]) }
    } else if (parts[0] == "KEY") {
        if (parts.size != 3) { throw IllegalArgumentException("invalid KEY line") }
        artifact.keys[parts[1]] = paperVector(parts[2])
    } else if (parts[0] == "TRAIN") {
        if (parts.size != 5 && parts.size != 7) { throw IllegalArgumentException("invalid TRAIN line") }
        let clusterKey = if (parts.size == 7) { parts[5] } else { "${parts[3]}-0001" }
        let clusterGroup = if (parts.size == 7) { parts[6] } else { parts[2] }
        artifact.records.add(PaperTrainRecord(
            parts[1], parts[2], parts[3], paperVector(parts[4]), clusterKey, clusterGroup
        ))
    } else if (parts[0] == "QUERY") {
        if (parts.size != 6) { throw IllegalArgumentException("invalid QUERY line") }
        artifact.queries.add(PaperQuery(parts[1], parts[2], parts[3], paperVector(parts[4]), paperVector(parts[5])))
    } else if (parts[0] == "GT") {
        if (parts.size != 4) { throw IllegalArgumentException("invalid GT line") }
        artifact.groundTruth.add(PaperGroundTruth(parts[1], Float64.parse(parts[2]), paperIds(parts[3])))
    } else {
        throw IllegalArgumentException("unknown paper artifact command: ${parts[0]}")
    }
}

func loadPaperArtifact(filePath: String): PaperArtifact {
    let artifact = PaperArtifact()
    let file = File(filePath, OpenMode.Read)
    let buffer = Array<Byte>(65536, repeat: 0)
    var carry = ""
    while (true) {
        let count = file.read(buffer)
        if (count <= 0) { break }
        carry += String.fromUtf8(Array<Byte>(count, { i => buffer[i] }))
        while (true) {
            match (carry.indexOf("\n")) {
                case Some(index) =>
                    parsePaperLine(carry[0..index], artifact)
                    carry = carry[(index + 1)..carry.size]
                case None => break
            }
        }
    }
    if (!carry.isEmpty()) { parsePaperLine(carry, artifact) }
    if (artifact.protocol != "python-paper-90-10") {
        throw IllegalArgumentException("expected python-paper-90-10 artifact, got ${artifact.protocol}")
    }
    if (artifact.records.isEmpty() || artifact.queries.isEmpty() || artifact.keys.isEmpty()) {
        throw IllegalArgumentException("paper artifact has empty records, queries or key vectors")
    }
    return artifact
}

func paperMean(vectors: ArrayList<Array<Float64>>): Array<Float64> {
    if (vectors.isEmpty()) { return Array<Float64>() }
    let output = Array<Float64>(vectors[0].size, repeat: 0.0)
    for (vector in vectors) {
        if (vector.size != output.size) { throw IllegalArgumentException("paper artifact dimension mismatch") }
        for (i in 0..output.size) { output[i] += vector[i] }
    }
    for (i in 0..output.size) { output[i] /= Float64(vectors.size) }
    return normalizedVector(output)
}

func paperExpected(artifact: PaperArtifact, queryId: String, beta: Float64): ArrayList<String> {
    for (row in artifact.groundTruth) {
        if (row.queryId == queryId && abs(row.beta - beta) < 0.000001) { return row.ids }
    }
    throw IllegalArgumentException("ground truth missing for query=${queryId}, beta=${beta}")
}

func paperOverlap(expected: ArrayList<String>, actual: ArrayList<String>, topK: Int64): Float64 {
    if (topK <= 0) { return 0.0 }
    var hits: Int64 = 0
    let limit = if (topK < actual.size) { topK } else { actual.size }
    for (i in 0..limit) { if (expected.contains(actual[i])) { hits++ } }
    return Float64(hits) / Float64(topK)
}

func paperMixedRelevance(artifact: PaperArtifact, query: PaperQuery,
                         recordId: String, beta: Float64): Float64 {
    for (record in artifact.records) {
        if (record.id != recordId) { continue }
        let embSimilarity = 1.0 - cosineDistanceNormalized(query.vector, record.vector)
        var entitySimilarity = 0.0
        if (artifact.keys.contains(record.key)) {
            entitySimilarity = 1.0 - cosineDistanceNormalized(query.keyVector, artifact.keys[record.key])
        }
        let score = beta * entitySimilarity + (1.0 - beta) * embSimilarity
        // Equation 5 defines both similarities on [0, 1]. Clamp tiny numeric
        // excursions so Equation 14 cannot receive an invalid negative gain.
        return clampUnit(score)
    }
    throw IllegalStateException("NDCG record is missing: ${recordId}")
}

// Equation 14 in the paper: mixed score is the graded gain signal. This is
// intentionally not binary hit-based NDCG.
func paperMixedNdcg(artifact: PaperArtifact, query: PaperQuery,
                    idealIds: ArrayList<String>, rankedIds: ArrayList<String>,
                    beta: Float64, topK: Int64): Float64 {
    if (topK <= 0 || idealIds.isEmpty()) { return 0.0 }
    let rankedLimit = if (topK < rankedIds.size) { topK } else { rankedIds.size }
    let idealLimit = if (topK < idealIds.size) { topK } else { idealIds.size }
    var dcg = 0.0
    var idcg = 0.0
    for (i in 0..rankedLimit) {
        let gain = pow(2.0, paperMixedRelevance(artifact, query, rankedIds[i], beta)) - 1.0
        dcg += gain / (log(Float64(i + 2)) / log(2.0))
    }
    for (i in 0..idealLimit) {
        let gain = pow(2.0, paperMixedRelevance(artifact, query, idealIds[i], beta)) - 1.0
        idcg += gain / (log(Float64(i + 2)) / log(2.0))
    }
    return if (idcg > 0.0) { dcg / idcg } else { 0.0 }
}

func paperRecordKeys(artifact: PaperArtifact, ids: ArrayList<String>, topK: Int64): ArrayList<String> {
    let keys = ArrayList<String>()
    let limit = if (topK < ids.size) { topK } else { ids.size }
    for (i in 0..limit) {
        var found = false
        for (record in artifact.records) {
            if (record.id == ids[i]) {
                keys.add(record.key)
                found = true
                break
            }
        }
        if (!found) { throw IllegalStateException("ground-truth record is missing: ${ids[i]}") }
    }
    return keys
}

func paperBeta(v: Float64): String {
    let scaled = Int64(v * 10.0 + 0.5)
    return "${scaled / 10}.${scaled % 10}"
}

func printPaperProgress(dataset: String, beta: Float64, completed: Int64, total: Int64): Unit {
    let width: Int64 = 30
    let filled = if (total > 0) { completed * width / total } else { width }
    let bar = "#" * filled + "-" * (width - filled)
    let percent = if (total > 0) { Float64(completed) * 100.0 / Float64(total) } else { 100.0 }
    println("PAPER_PROGRESS|dataset=${dataset}|beta=${paperBeta(beta)}|query=${completed}|total=${total}|percent=${formatFloat2(percent)}|[${bar}] ${completed}/${total}")
}

func paperTotalSeconds(stats: MethodRecallStats): Float64 {
    return Float64(stats.totalLatencyNs) / 1000000000.0
}

func printPaperPythonStyle(
    stats: BetaEvalStats,
    groundTruth: MethodRecallStats,
    topK: Int64,
    totalQueries: Int64
): Unit {
    let beta = paperBeta(stats.beta)
    println("\n" + "=" * 80)
    println("=== Recall Test results ===")
    println("=" * 80)
    println("Total queries: ${totalQueries}")
    println("Number of successful queries: ${totalQueries}")
    println("\n--- Recall Indicator header ---")
    println("| Method                | Avg Recall@${topK} | Min Recall | Max Recall |")
    println("|-----------------------|-------------------|------------|-----------|")
    println("| ${padRight("Mixed (beta=${beta}, HDMG)", 21)} | ${formatFloat4(stats.mixed.avg())}         | ${formatFloat4(stats.mixed.minVal())}   | ${formatFloat4(stats.mixed.maxVal())}   |")
    println("| ${padRight("w/o HDMG", 21)} | ${formatFloat4(stats.withoutHdmg.avg())}         | ${formatFloat4(stats.withoutHdmg.minVal())}   | ${formatFloat4(stats.withoutHdmg.maxVal())}   |")
    println("| ${padRight("Representative", 21)} | ${formatFloat4(stats.representative.avg())}         | ${formatFloat4(stats.representative.minVal())}   | ${formatFloat4(stats.representative.maxVal())}   |")
    println("| ${padRight("PythonFlat", 21)} | ${formatFloat4(stats.pyflat.avg())}         | ${formatFloat4(stats.pyflat.minVal())}   | ${formatFloat4(stats.pyflat.maxVal())}   |")

    println("\n--- Time comparison ---")
    println("Ground Truth (brute-force mixed, beta=${beta}) Total time: ${formatFloat2(paperTotalSeconds(groundTruth))} (${formatFloat2(paperTotalSeconds(groundTruth) / 60.0)} )")
    println("Ground Truth Average query: ${formatFloat2(groundTruth.avgLatencyMs())} millisecond")
    println("Mixed Total search time: ${formatFloat2(paperTotalSeconds(stats.mixed))} (${formatFloat2(paperTotalSeconds(stats.mixed) / 60.0)} )")
    println("Mixed Average query: ${formatFloat2(stats.mixed.avgLatencyMs())} millisecond")
    println("w/o HDMG Total search time: ${formatFloat2(paperTotalSeconds(stats.withoutHdmg))} (${formatFloat2(paperTotalSeconds(stats.withoutHdmg) / 60.0)} )")
    println("w/o HDMG Average query: ${formatFloat2(stats.withoutHdmg.avgLatencyMs())} millisecond")
    println("Representative Total search time: ${formatFloat2(paperTotalSeconds(stats.representative))} (${formatFloat2(paperTotalSeconds(stats.representative) / 60.0)} )")
    println("Representative Average query: ${formatFloat2(stats.representative.avgLatencyMs())} millisecond")
    printHdmgTraceStats(stats.hdmgTrace)
    if (stats.hdmgTrace.queries > 0) {
        println("HDMG Average time taken: entrance selection=${formatMs(stats.hdmgTrace.avgEntryMs())}ms, Figure wandering=${formatMs(stats.hdmgTrace.avgWalkMs())}ms, Reorder=${formatMs(stats.hdmgTrace.avgRerankMs())}ms")
    }
    println("PythonFlat Total search time: ${formatFloat2(paperTotalSeconds(stats.pyflat))} (${formatFloat2(paperTotalSeconds(stats.pyflat) / 60.0)} )")
    println("PythonFlat Average query: ${formatFloat2(stats.pyflat.avgLatencyMs())} millisecond")
    if (stats.mixed.totalLatencyNs > 0) {
        let speedup = Float64(groundTruth.totalLatencyNs) / Float64(stats.mixed.totalLatencyNs)
        println("\nMixed Speedup ratio: ${formatFloat2(speedup)}x (Faster than Ground Truth ${formatFloat2(speedup)} )")
    }
    if (stats.withoutHdmg.totalLatencyNs > 0) {
        let speedup = Float64(stats.withoutHdmg.totalLatencyNs) / Float64(stats.mixed.totalLatencyNs)
        println("HDMG vs w/o HDMG Speedup ratio: ${formatFloat2(speedup)}x")
    }
    if (stats.representative.totalLatencyNs > 0) {
        let speedup = Float64(groundTruth.totalLatencyNs) / Float64(stats.representative.totalLatencyNs)
        println("Representative Speedup ratio: ${formatFloat2(speedup)}x (Faster than Ground Truth ${formatFloat2(speedup)} )")
    }
    if (stats.pyflat.totalLatencyNs > 0) {
        let speedup = Float64(groundTruth.totalLatencyNs) / Float64(stats.pyflat.totalLatencyNs)
        println("PythonFlat Speedup ratio: ${formatFloat2(speedup)}x (Faster than Ground Truth ${formatFloat2(speedup)} )")
    }
    println("=" * 80)
}

class PaperRuntime {
    let artifact: PaperArtifact
    let vectormap: VectorMap
    let flatItems: ArrayList<FlatItem>
    let searchConfig: HdmgSearchConfig

    init(artifact: PaperArtifact, vectormap: VectorMap, flatItems: ArrayList<FlatItem>,
         searchConfig: HdmgSearchConfig) {
        this.artifact = artifact
        this.vectormap = vectormap
        this.flatItems = flatItems
        this.searchConfig = searchConfig
    }
}

func buildPaperRuntime(filePath: String): PaperRuntime {
    var stageStart = DateTime.now().toUnixTimeStamp().toNanoseconds()
    println("PAPER_STAGE|load|start|path=${filePath}")
    let artifact = loadPaperArtifact(filePath)
    var stageEnd = DateTime.now().toUnixTimeStamp().toNanoseconds()
    println("PAPER_STAGE|load|done|ms=${Float64(stageEnd - stageStart) / 1000000.0}|training=${artifact.records.size}|queries=${artifact.queries.size}")
    stageStart = stageEnd
    let byCluster = HashMap<String, ArrayList<PaperTrainRecord>>()
    for (record in artifact.records) {
        if (!byCluster.contains(record.clusterKey)) {
            byCluster[record.clusterKey] = ArrayList<PaperTrainRecord>()
        }
        byCluster[record.clusterKey].add(record)
    }
    let vectormap = VectorMap()
    let flatItems = ArrayList<FlatItem>()
    for ((clusterKey, records) in byCluster) {
        let vectors = ArrayList<Array<Float64>>()
        for (record in records) { vectors.add(record.vector) }
        let descriptions = Array<Any>(records.size, { i => records[i].id })
        let group = VectorGroup(records[0].clusterGroup, paperMean(vectors), "simple",
                                vectors.toArray(), descriptions,
                                vectorType: "image", groupType: "")
        // Python stores every micro-cluster under its semantic base key and
        // distinguishes clusters by group_name.  Keeping clusterKey here made
        // Representative/w/o-HDMG look up keys such as "Faces-0014", while
        // the frozen key-vector table contains "Faces"; semantic distance was
        // therefore silently omitted from those searches.
        vectormap.insert(records[0].key, group, metadataOpt: None)
        for (record in records) { flatItems.add(FlatItem(record.id, record.key, record.vector)) }
    }
    stageEnd = DateTime.now().toUnixTimeStamp().toNanoseconds()
    println("PAPER_STAGE|groups|done|ms=${Float64(stageEnd - stageStart) / 1000000.0}|groups=${byCluster.size}")
    stageStart = stageEnd
    vectormap.setKeyVectors(artifact.keys)
    // These are the defaults used by caltech_bench.py/cub_bench.py/coco_bench.py.
    vectormap.buildHdmgWithConfig(HdmgBuildConfig(
        embeddingK: 12,
        semanticIntraK: 20,
        semanticBridgeKeys: 2,
        semanticBridgePerKey: 1,
        useMutualEmbedding: false,
        assumeNormalized: true
    ))
    stageEnd = DateTime.now().toUnixTimeStamp().toNanoseconds()
    println("PAPER_STAGE|hdmg|done|ms=${Float64(stageEnd - stageStart) / 1000000.0}")
    let searchConfig = HdmgSearchConfig(
        maxSteps: 100,
        entryAlphaThreshold: 0.3,
        clusterPoolMultiplier: 3,
        topKeyCandidates: 5,
        extraHops: 0
    )
    return PaperRuntime(artifact, vectormap, flatItems, searchConfig)
}

func evaluatePaperBeta(runtime: PaperRuntime, maxQueries: Int64, beta: Float64):
    (BetaEvalStats, MethodRecallStats, Int64) {
    let artifact = runtime.artifact
    let vectormap = runtime.vectormap
    let flatItems = runtime.flatItems
    println("\n" + "=" * 80)
    println("=== Recall test ===")
    println("Ground Truth: brute-force mixed (beta=${paperBeta(beta)}) Search top ${artifact.topK}")
    println("Mixed (beta=${paperBeta(beta)}, HDMG): Mixed key+rep vector search top ${artifact.topK}")
    println("w/o HDMG: route top-5 entity groups, directly scan up to ${artifact.topK * 3} micro-clusters per group, then mixed rerank top ${artifact.topK}")
    println("Representative-3 (supplementary): globally select top-${artifact.topK} micro-clusters by mixed representative score, then rerank top ${artifact.topK}")
    println("PythonFlat: Flat + Cosine Search top ${artifact.topK}")
    println("=" * 80)
    println("test all ${if (maxQueries < artifact.queries.size) { maxQueries } else { artifact.queries.size }} query")

    let stats = BetaEvalStats(beta)
    let groundTruthStats = MethodRecallStats("Ground Truth")
    var count: Int64 = 0
    let totalQueries = if (maxQueries < artifact.queries.size) { maxQueries } else { artifact.queries.size }
    let useKeyRecall = beta > 0.999999
    printPaperProgress(artifact.dataset, beta, 0, totalQueries)
    for (query in artifact.queries) {
        if (count >= maxQueries) { break }
        let expectedMixed = paperExpected(artifact, query.id, beta)

        // Match Python timing boundaries: each timer wraps only one search call.
        // Loading, graph construction, metric calculation and printing stay outside.
        let gtStart = nowNs()
        let groundTruth = flatMixedSearch(
            flatItems, query.vector, Some(query.keyVector), artifact.keys, beta, artifact.topK,
            assumeNormalized: true
        )
        groundTruthStats.addLatency(nowNs() - gtStart)

        let mixedStart = nowNs()
        let mixed = vectormap.searchHdmgWithConfig(
            query.vector, Some(query.keyVector), beta, artifact.topK, runtime.searchConfig,
            distanceMethod: "cosine_normalized"
        )
        stats.mixed.addLatency(nowNs() - mixedStart)
        recordHdmgTrace(vectormap, stats.hdmgTrace)

        // Paper Section 6.1 ablation: retain entity routing and mixed scoring,
        // but directly scan micro-clusters inside the routed groups.
        let withoutHdmgStart = nowNs()
        let withoutHdmg = vectormap.searchWithoutHdmg(
            query.vector,
            queryKeyVector: Some(query.keyVector),
            beta: beta,
            topK: artifact.topK,
            groupBudget: runtime.searchConfig.topKeyCandidates,
            microClusterBudget: artifact.topK * runtime.searchConfig.clusterPoolMultiplier,
            distanceMethod: "cosine_normalized"
        )
        stats.withoutHdmg.addLatency(nowNs() - withoutHdmgStart)

        let representativeStart = nowNs()
        let representative = vectormap.searchWithRepresentativeRerank(
            query.vector,
            queryKeyVector: Some(query.keyVector),
            beta: beta,
            topK: artifact.topK,
            numGroups: Some(artifact.topK),
            distanceMethod: "cosine_normalized"
        )
        stats.representative.addLatency(nowNs() - representativeStart)

        let pythonFlatStart = nowNs()
        let pythonFlat = flatEmbeddingSearch(flatItems, query.vector, artifact.topK, assumeNormalized: true)
        stats.pyflat.addLatency(nowNs() - pythonFlatStart)

        let gtIds = ArrayList<String>()
        for (row in groundTruth) { gtIds.add(row.id) }
        let mixedIds = searchResultIds(mixed, artifact.topK)
        let withoutHdmgIds = searchResultIds(withoutHdmg, artifact.topK)
        let representativeIds = searchResultIds(representative, artifact.topK)
        let pythonFlatIds = ArrayList<String>()
        for (row in pythonFlat) { pythonFlatIds.add(row.id) }
        if (useKeyRecall) {
            let gtKeys = flatBaseKeys(groundTruth, artifact.topK)
            let expectedKeys = paperRecordKeys(artifact, expectedMixed, artifact.topK)
            // At beta=1 every record under the same semantic key has exactly
            // the same score. Python and Cangjie may therefore choose different
            // record IDs from the tie; validate the semantic keys, which is also
            // how the paper benchmark scores this endpoint.
            if (recallKeyHitRate(expectedKeys, gtKeys, artifact.topK) < 0.999999) {
                throw IllegalStateException("frozen ground-truth keys differ from Cangjie brute-force result for query=${query.id}, beta=${beta}")
            }
            let mixedKeys = resultBaseKeys(mixed, artifact.topK)
            let withoutHdmgKeys = resultBaseKeys(withoutHdmg, artifact.topK)
            let representativeKeys = resultBaseKeys(representative, artifact.topK)
            let pythonFlatKeys = flatBaseKeys(pythonFlat, artifact.topK)
            stats.mixed.recalls.add(recallKeyHitRate(gtKeys, mixedKeys, artifact.topK))
            stats.withoutHdmg.recalls.add(recallKeyHitRate(gtKeys, withoutHdmgKeys, artifact.topK))
            stats.representative.recalls.add(recallKeyHitRate(gtKeys, representativeKeys, artifact.topK))
            stats.pyflat.recalls.add(recallKeyHitRate(gtKeys, pythonFlatKeys, artifact.topK))
            stats.mixed.ndcgs.add(paperMixedNdcg(artifact, query, gtIds, mixedIds, beta, artifact.topK))
            stats.withoutHdmg.ndcgs.add(paperMixedNdcg(artifact, query, gtIds, withoutHdmgIds, beta, artifact.topK))
            stats.representative.ndcgs.add(paperMixedNdcg(artifact, query, gtIds, representativeIds, beta, artifact.topK))
            stats.pyflat.ndcgs.add(paperMixedNdcg(artifact, query, gtIds, pythonFlatIds, beta, artifact.topK))
        } else {
            if (paperOverlap(expectedMixed, gtIds, artifact.topK) < 0.999999) {
                throw IllegalStateException("frozen ground truth differs from Cangjie brute-force result for query=${query.id}, beta=${beta}")
            }
            stats.mixed.recalls.add(paperOverlap(gtIds, mixedIds, artifact.topK))
            stats.withoutHdmg.recalls.add(paperOverlap(gtIds, withoutHdmgIds, artifact.topK))
            stats.representative.recalls.add(paperOverlap(gtIds, representativeIds, artifact.topK))
            stats.pyflat.recalls.add(paperOverlap(gtIds, pythonFlatIds, artifact.topK))
            stats.mixed.ndcgs.add(paperMixedNdcg(artifact, query, gtIds, mixedIds, beta, artifact.topK))
            stats.withoutHdmg.ndcgs.add(paperMixedNdcg(artifact, query, gtIds, withoutHdmgIds, beta, artifact.topK))
            stats.representative.ndcgs.add(paperMixedNdcg(artifact, query, gtIds, representativeIds, beta, artifact.topK))
            stats.pyflat.ndcgs.add(paperMixedNdcg(artifact, query, gtIds, pythonFlatIds, beta, artifact.topK))
        }
        if (maxQueries <= 10) {
            println("PAPER_QUERY|${query.id}|groundTruth=${parityFlatTrace(groundTruth, artifact.topK)}|mixed=${parityResultTrace(mixed, artifact.topK)}|withoutHdmg=${parityResultTrace(withoutHdmg, artifact.topK)}|representative=${parityResultTrace(representative, artifact.topK)}|pythonFlat=${parityFlatTrace(pythonFlat, artifact.topK)}")
        }
        count++
        // Keep this outside every search timer so progress reporting never changes
        // the latency figures. Direct terminal runs show one exact update per query.
        printPaperProgress(artifact.dataset, beta, count, totalQueries)
    }
    printPaperPythonStyle(stats, groundTruthStats, artifact.topK, count)
    return (stats, groundTruthStats, count)
}

func paperSummaryLine(runtime: PaperRuntime, stats: BetaEvalStats,
                      groundTruthStats: MethodRecallStats, count: Int64): String {
    let artifact = runtime.artifact
    let beta = stats.beta
    let queryScope = if (artifact.queryPoolSize > 0 && artifact.queries.size == artifact.queryPoolSize && count == artifact.queryPoolSize) {
        "full-10-percent-test-pool"
    } else {
        "debug-subset"
    }
    return "PAPER_SUMMARY|protocol=${artifact.protocol}|evaluationProtocol=violas-paper-table2-v5|queryScope=${queryScope}|implementation=normalized-f64-o2|ndcgGain=mixed-score-graded|withoutHdmgMethod=entity-route-plus-local-microcluster-scan|withoutHdmgGroupBudget=${runtime.searchConfig.topKeyCandidates}|withoutHdmgMicroClusterBudget=${artifact.topK * runtime.searchConfig.clusterPoolMultiplier}|dataset=${artifact.dataset}|training=${artifact.records.size}|queries=${count}|beta=${beta}|mixedRecall=${stats.mixed.avg()}|mixedNdcg=${stats.mixed.avgNdcg()}|mixedMin=${stats.mixed.minVal()}|mixedMax=${stats.mixed.maxVal()}|withoutHdmgRecall=${stats.withoutHdmg.avg()}|withoutHdmgNdcg=${stats.withoutHdmg.avgNdcg()}|withoutHdmgMin=${stats.withoutHdmg.minVal()}|withoutHdmgMax=${stats.withoutHdmg.maxVal()}|representativeRecall=${stats.representative.avg()}|representativeNdcg=${stats.representative.avgNdcg()}|representativeMin=${stats.representative.minVal()}|representativeMax=${stats.representative.maxVal()}|pythonFlatRecall=${stats.pyflat.avg()}|pythonFlatNdcg=${stats.pyflat.avgNdcg()}|mixedLatencyMs=${stats.mixed.avgLatencyMs()}|withoutHdmgLatencyMs=${stats.withoutHdmg.avgLatencyMs()}|representativeLatencyMs=${stats.representative.avgLatencyMs()}|pythonFlatLatencyMs=${stats.pyflat.avgLatencyMs()}|groundTruthLatencyMs=${groundTruthStats.avgLatencyMs()}|hdmgHops=${stats.hdmgTrace.avgHops()}|hdmgDepth=${stats.hdmgTrace.avgDepth()}|hdmgCandidatePool=${stats.hdmgTrace.avgCandidatePool()}|hdmgEntryMs=${stats.hdmgTrace.avgEntryMs()}|hdmgWalkMs=${stats.hdmgTrace.avgWalkMs()}|hdmgRerankMs=${stats.hdmgTrace.avgRerankMs()}"
}

func printAndPersistPaperSummary(runtime: PaperRuntime, stats: BetaEvalStats,
                                 groundTruthStats: MethodRecallStats, count: Int64,
                                 summaryPath: String): Unit {
    let line = paperSummaryLine(runtime, stats, groundTruthStats, count)
    println(line)
    if (!summaryPath.isEmpty()) {
        File.appendTo(summaryPath, (line + "\n").toArray())
    }
}

func printPaperSweepSummary(allStats: ArrayList<BetaEvalStats>): Unit {
    println("\n" + "=" * 140)
    println("=== (alpha=0.5) ===")
    println("=" * 140)
    println("exp_factor beta | mixed_recall min  max  | w/o_hdmg   min  max  | representative min  max  | pyflat   milvus   milvus_mix qdrant  qdrant_mix chroma   chroma_mix")
    println("-" * 140)
    for (stats in allStats) {
        println("3          ${paperBeta(stats.beta)}  | ${formatFloat4(stats.mixed.avg())}       ${formatFloat2(stats.mixed.minVal())} ${formatFloat2(stats.mixed.maxVal())} | ${formatFloat4(stats.withoutHdmg.avg())}       ${formatFloat2(stats.withoutHdmg.minVal())} ${formatFloat2(stats.withoutHdmg.maxVal())} | ${formatFloat4(stats.representative.avg())}       ${formatFloat2(stats.representative.minVal())} ${formatFloat2(stats.representative.maxVal())} | ${formatFloat4(stats.pyflat.avg())} N/A      N/A        N/A     N/A        N/A      N/A")
    }
    println("=" * 140)

    println("\n" + "=" * 140)
    println("=== (ms/query) ===")
    println("=" * 140)
    println("exp_factor beta | hdmg       w/o_hdmg   representative pyflat     milvus     milvus_mix qdrant    qdrant_mix chroma     chroma_mix")
    println("-" * 140)
    for (stats in allStats) {
        println("3          ${paperBeta(stats.beta)}  | ${padRight(formatFloat2(stats.mixed.avgLatencyMs()), 10)} ${padRight(formatFloat2(stats.withoutHdmg.avgLatencyMs()), 10)} ${padRight(formatFloat2(stats.representative.avgLatencyMs()), 14)} ${padRight(formatFloat2(stats.pyflat.avgLatencyMs()), 10)} N/A        N/A        N/A       N/A        N/A        N/A")
    }
    println("=" * 140)

    println("\n" + "=" * 120)
    println("=== HDMG (beta) ===")
    println("=" * 120)
    println("beta  | avg_hops   avg_depth  cand_pool  ms_calls   | t_entry    t_walk     t_rerank   t_total")
    println("-" * 120)
    for (stats in allStats) {
        let trace = stats.hdmgTrace
        let total = trace.avgEntryMs() + trace.avgWalkMs() + trace.avgRerankMs()
        println("${paperBeta(stats.beta)}   | ${padRight(formatFloat2(trace.avgHops()), 10)} ${padRight(formatFloat2(trace.avgDepth()), 10)} ${padRight(formatFloat2(trace.avgCandidatePool()), 10)} ${padRight(formatFloat2(trace.avgMixedScoreCalls()), 10)} | ${padRight(formatMs(trace.avgEntryMs()), 10)} ${padRight(formatMs(trace.avgWalkMs()), 10)} ${padRight(formatMs(trace.avgRerankMs()), 10)} ${formatMs(total)}")
    }
    println("=" * 120)
}

public func runPaperArtifact(filePath: String, maxQueries: Int64, beta: Float64,
                             summaryPath!: String = ""): Unit {
    if (maxQueries < 0 || beta < 0.0 || beta > 1.0) {
        throw IllegalArgumentException("paper maxQueries must be non-negative and beta must be in [0,1]")
    }
    if (!summaryPath.isEmpty()) { File.writeTo(summaryPath, "".toArray()) }
    let runtime = buildPaperRuntime(filePath)
    let queryLimit = if (maxQueries == 0) { runtime.artifact.queries.size } else { maxQueries }
    let (stats, groundTruthStats, count) = evaluatePaperBeta(runtime, queryLimit, beta)
    printAndPersistPaperSummary(runtime, stats, groundTruthStats, count, summaryPath)
}

public func runPaperArtifactSweep(filePath: String, maxQueries: Int64,
                                  summaryPath!: String = ""): Unit {
    if (maxQueries < 0) {
        throw IllegalArgumentException("paper maxQueries must be non-negative")
    }
    if (!summaryPath.isEmpty()) { File.writeTo(summaryPath, "".toArray()) }
    let runtime = buildPaperRuntime(filePath)
    let queryLimit = if (maxQueries == 0) { runtime.artifact.queries.size } else { maxQueries }
    let allStats = ArrayList<BetaEvalStats>()
    println("\n" + "=" * 80)
    println("=== Multiple sets of parameters Recall test ===")
    println("Testing 11 parameter combinations")
    println("alpha: 0.5")
    println("hdmg_cluster_pool_size: 9")
    println("=" * 80)
    for (index in 0..11) {
        let beta = Float64(index) / 10.0
        println("\n--- No. ${index + 1}/11 Group parameters ---")
        println("beta: ${paperBeta(beta)}")
        let (stats, groundTruthStats, count) = evaluatePaperBeta(runtime, queryLimit, beta)
        allStats.add(stats)
        printAndPersistPaperSummary(runtime, stats, groundTruthStats, count, summaryPath)
    }
    printPaperSweepSummary(allStats)
}

func paperMaintenanceVector(vector: Array<Float64>, offset: Int64): Array<Float64> {
    let size = vector.size
    if (size == 0) { return vector }
    let shift = 1 + offset % 7
    return normalizedVector(Array<Float64>(size, { i =>
        0.99 * vector[i] + 0.01 * vector[(i + size - shift) % size]
    }))
}

/** 使用稳定 record ID 评测对象 CRUD 与 HDMG 完整重建。 */
public func runPaperMaintenance(
    filePath: String,
    requestedCount: Int64,
    summaryPath!: String = ""
): Unit {
    if (requestedCount <= 0) {
        throw IllegalArgumentException("maintenance count must be positive")
    }
    let runtime = buildPaperRuntime(filePath)
    let artifact = runtime.artifact
    let count = min(requestedCount, min(artifact.records.size, artifact.queries.size))
    let initialBuildNs =
        (runtime.vectormap.getLastHdmgSearchStats()["t_build_ns"] as Int64).getOrThrow()

    var start = nowNs()
    for (i in 0..count) {
        let query = artifact.queries[i]
        if (!runtime.vectormap.data.contains(query.trueKey) ||
                runtime.vectormap.data[query.trueKey].groups.isEmpty()) {
            throw IllegalStateException("no target group for inserted query ${query.id}")
        }
        let groupName = runtime.vectormap.data[query.trueKey].groups[0].groupName
        if (!runtime.vectormap.insertObject(
                query.trueKey, groupName, query.sourceId, query.vector)) {
            throw IllegalStateException("failed to insert ${query.sourceId}")
        }
    }
    let insertionNs = nowNs() - start
    start = nowNs()
    runtime.vectormap.buildHdmgWithConfig(HdmgBuildConfig(
        embeddingK: 12, semanticIntraK: 20, semanticBridgeKeys: 2,
        semanticBridgePerKey: 1, useMutualEmbedding: false, assumeNormalized: true
    ))
    let insertionIndexNs = nowNs() - start

    start = nowNs()
    for (i in 0..count) {
        let record = artifact.records[i]
        if (!runtime.vectormap.updateObject(
                record.id, paperMaintenanceVector(record.vector, i))) {
            throw IllegalStateException("failed to update ${record.id}")
        }
    }
    let updateNs = nowNs() - start
    start = nowNs()
    runtime.vectormap.buildHdmgWithConfig(HdmgBuildConfig(
        embeddingK: 12, semanticIntraK: 20, semanticBridgeKeys: 2,
        semanticBridgePerKey: 1, useMutualEmbedding: false, assumeNormalized: true
    ))
    let updateIndexNs = nowNs() - start

    start = nowNs()
    for (i in 0..count) {
        let sourceId = artifact.queries[i].sourceId
        if (!runtime.vectormap.deleteObject(sourceId)) {
            throw IllegalStateException("failed to delete ${sourceId}")
        }
    }
    let deleteNs = nowNs() - start
    start = nowNs()
    runtime.vectormap.buildHdmgWithConfig(HdmgBuildConfig(
        embeddingK: 12, semanticIntraK: 20, semanticBridgeKeys: 2,
        semanticBridgePerKey: 1, useMutualEmbedding: false, assumeNormalized: true
    ))
    let deleteIndexNs = nowNs() - start

    let line = "MAINTENANCE_SUMMARY|backend=cangjie-violas|index=HDMG" +
        "|dataset=${artifact.dataset}|initialRecords=${artifact.records.size}|count=${count}" +
        "|vectorInsertionMs=${Float64(insertionNs) / 1000000.0}" +
        "|vectorUpdateMs=${Float64(updateNs) / 1000000.0}" +
        "|vectorDeleteMs=${Float64(deleteNs) / 1000000.0}" +
        "|indexConstructionMs=${Float64(initialBuildNs) / 1000000.0}" +
        "|indexAfterInsertMs=${Float64(insertionIndexNs) / 1000000.0}" +
        "|indexAfterUpdateMs=${Float64(updateIndexNs) / 1000000.0}" +
        "|indexAfterDeleteMs=${Float64(deleteIndexNs) / 1000000.0}" +
        "|maintenanceStrategy=stable-id-in-place-crud-plus-full-hdmg-rebuild"
    println(line)
    if (!summaryPath.isEmpty()) {
        File.writeTo(summaryPath, (line + "\n").toArray())
    }
}
