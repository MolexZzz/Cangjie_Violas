package semanticvg_cj.examples

import std.collection.*
import std.random.*
import semanticvg_cj.storage.*

public func runMinimalVectorMapDemo(): Unit {
    let vectormap = VectorMap()

    let rhino = makeSyntheticGroup("rhino", [1.0, 0.0, 0.0, 0.0], 6)
    let duck = makeSyntheticGroup("duck", [0.0, 1.0, 0.0, 0.0], 6)
    let drug = makeSyntheticGroup("norfloxacin", [0.0, 0.0, 1.0, 0.0], 6)

    vectormap.insert("rhino", rhino, metadataOpt: None)
    vectormap.insert("duck", duck, metadataOpt: None)
    vectormap.insert("norfloxacin", drug, metadataOpt: None)

    let query = [0.95, 0.05, 0.0, 0.0]
    let results = vectormap.search(query, topK: 3, mode: "single", distanceMethod: "cosine")

    var rank = 1
    for (r in results) {
        let idx = r.vectorIdx.getOrThrow()
        let desc = r.group.descriptions[idx]
        let textOpt = desc["text"] as String
        let text = if (textOpt.isSome()) { textOpt.getOrThrow() } else { "" }
        println("${rank}. key=${r.key} group=${r.group.groupName} distance=${r.distance} text=${text}")
        rank++
    }
}

func makeSyntheticGroup(name: String, center: Array<Float64>, count: Int64): VectorGroup {
    let rng = Random()
    let vectors = ArrayList<Array<Float64>>()
    let descs = ArrayList<Any>()

    var i = 0
    while (i < count) {
        let noise = Array<Float64>(center.size, { _ => (rng.nextFloat64() - 0.5) * 0.08 })
        let vec = Array<Float64>(center.size, { j => center[j] + noise[j] })
        vectors.add(vec)
        descs.add("{\"text\": \"${name} item ${i}\"}")
        i++
    }

    let vecArr = Array<Array<Float64>>(vectors.size, { k => vectors[k] })
    let descArr = Array<Any>(descs.size, { k => descs[k] })

    var rep = Array<Float64>(center.size, { _ => 0.0 })
    for (v in vecArr) {
        for (j in 0..v.size) { rep[j] += v[j] }
    }
    for (j in 0..rep.size) { rep[j] /= Float64(vecArr.size) }

    return VectorGroup(
        name,
        rep,
        "${name} representative",
        vecArr,
        descArr,
        vectorType: "demo",
        groupType: "synthetic"
    )
}
