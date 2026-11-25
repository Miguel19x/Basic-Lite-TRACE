# Basic-Lite TRACE — Manifiesto de Versión y Criptografía (RELEASE_MANIFEST)

**Proyecto:** Basic-Lite TRACE  
**Versión Documentada:** TRACE 1.5.0 (Fase 4A Motor C 2B AVX-512 VNNI + GQA + BPE 128K; Fase 4B Banco 4 Dominios + TIR Sandbox + SQLite L3 + PDA JSON C)  
**Fecha de corte:** 25 de noviembre de 2026  
**Titular y Custodio:** Miguel19x (`miguelaranguren19x@outlook.com`)  

> [!NOTE]
> Este manifiesto sirve para identificar fehacientemente la composición e integridad de los archivos del paquete en la fecha especificada mediante huellas criptográficas SHA-256. No constituye por sí solo una certificación legal o notarial de prioridad, sino una prueba técnica inmutable de existencia previa e integridad (*prior art / proof of non-alteration*). El sello externo previo `docs/RELEASE_MANIFEST.md.tsr` corresponde a la instantánea v1.3.0; tras esta actualización v1.5.0 puede emitirse un nuevo sello RFC 3161 sobre este manifiesto.

---

## 1. Entorno Físico y de Ejecución Documentado

* **Procesador (CPU):** Intel Core i3-1115G4 @ 3.00 GHz base (2 núcleos físicos / 4 hilos lógicos, AVX-512 VNNI / AVX2 / FMA / F16C, 6 MB SmartCache L3).
* **Memoria RAM:** 16 GB DDR4 @ 3200 MT/s.
* **Almacenamiento secundario:** SSD NVMe PCIe Gen4 x4 (MSI M450 500GB).
* **Sistema Operativo:** Linux x86_64 (Kernel 6.x).
* **Herramientas de compilación y runtime:** GCC 13+ con soporte OpenMP, AVX2 y AVX-512 VNNI (`-Wall -Wextra -Werror -Wvla`), Python 3.11+, PyTorch 2.x (CPU-only build).

---

## 2. Hashes Criptográficos SHA-256 de Archivos Fuente y Artefactos Clave (Versión Consolidada v1.5.0)

Los siguientes hashes han sido generados mediante el algoritmo estándar SHA-256 sobre el contenido exacto de los archivos consolidados de la versión 1.5.0:

| Archivo Fuente / Artefacto | Hash Criptográfico SHA-256 |
| :--- | :--- |
| `README.md` | `d10ae6dec1d09b3114d948dbfe1df8d51b74c9d1877d9f471e1a005133715753` |
| `README.es.md` | `256efe6f7f60ff8a0531e7ebfb7916b2aeca33335c61d4b457c8c56f85929bcc` |
| `LICENSE.txt` | `933abb675562927cd166d6130ee20dbdb5545613c9122019305cb05f44613143` |
| `LICENSE-EN.md` | `06827a519c33975b2e2ebc9e7f20342b8ff42c878166947fdbb0977b23d3f241` |
| `CONTRIBUTING.md` | `ae4af9dadeec6125b5600cccc69888b6cc8dec7e569067d64f05a10819af153f` |
| `CODE_OF_CONDUCT.md` | `31a86da1c334e7c5c79503392d135456356c2ec181000f8466eca270db264f6d` |
| `SECURITY.md` | `1ba029741067a5c87758b8f16103f123362dffbb4d5d0af1a26f6d6b969a34a1` |
| `NOTICE.md` | `f95611c431e750afb376053709a3fb897e51b1ff04bd4004ef4906f20b754d36` |
| `AUTHORS.md` | `6adb579960af5cffb5b00d33c6cf3f5ce46ad2be1b016d249c1204ea75dcf5b7` |
| `CHANGELOG.md` | `0003b51548299d8a2122ae2d9140646d5962a8ead8d4606a7065bf8dc416315e` |
| `THIRD_PARTY_NOTICES.md` | `16bd24ef38f4be7aece90b0031260443eaa7e21360ce82546116c930398283a0` |
| `Makefile` | `b848f324de688c7bd88490a7d60f29bf99bc85039def719c45b6be0daebc40bb` |
| `config.py` | `2633f978f5e01bf2456f22ed6e741f1afb4394b602924da406cd5c10d1532925` |
| `train.py` | `5063458ebf853a4182692d927be9b845dcab35c2400ce4f2fb640e35def75946` |
| `eval.py` | `b8dc7b07303df47f64d3209f9d094a54e4807102560911651aba166b597f3cff` |
| `data.py` | `6e88a400f637c0120c258bbe4bf03b671d0dfab7bf6a064d522bbdf053db37d3` |
| `export_paged.py` | `a833f52adefbe59ea628b3f84bdaf7170949e3aa5a376f828ac8bb34bfe6dc9d` |
| `export_paged_lm.py` | `949b429c0d0d698becdc8e1a2bfd201d1df644c5c3b5eb0bc3a6a8dc8690968e` |
| `eval_fase1_5semillas.py` | `fcb980df3d3d0c6ebdb85c6a8af93d8e60001da07239bc63e27ef8aba272db18` |
| `csrc/paged_dispatch.c` | `df592f6be7a95a9d6ca855558a573f03cecbb4e98835167db40d16ab434a44e2` |
| `csrc/paged_dispatch.h` | `22b378ad3d1e5f1cb8acc2a93637f82a1b8877d27d6e125535fdfce2bef5d271` |
| `csrc/paged_gemm_avx512.c` | `f7f6bafb1a4cac19cc50889fdb3a851b4119e890595d59071edd67febf49cfcb` |
| `csrc/paged_gemm_avx2.c` | `3ad5b1f9bc4fd44fc0a969904d636b8b41f4e06105305c21c7bde6e705ab53bd` |
| `csrc/paged_gemm_scalar.c` | `c7a4b875e56f71ebd5a6ae77549a9eb724df4ec11ea69f0e89c6ead1e7ac7f04` |
| `csrc/paged_unpack.h` | `4a6ddef34f3f56e54e62ef1e7442b73539e2ebe622d1a79be30b2e6994933864` |
| `csrc/trace_causal.c` | `6507de7e8653733b660960598a902be6631c6cd9bd20a8e77a49f84c1bf7873a` |
| `csrc/trace_infer.h` | `ec380279b44c3eed25387aeeba9513ac5d819723f73352d65bf1db305597465b` |
| `csrc/trace_loader.c` | `5b675ba3cbdd38c936d5f1b6448a90a036ba7b70b8c1958efd5d76d1da124c28` |
| `csrc/trace_math.h` | `f2e5ab0cac4b34554eb597c78c15e639eb522942a0cf7a7b317097c7fc7ca7a2` |
| `csrc/lm/trace_lm.h` | `fd1955ff89d8fe36547d3cd62171068f41d91f06f883e53c5befa0e9c429c05d` |
| `csrc/lm/trace_lm_cli.c` | `ec20b25c9ed3a0b79dd89e3ce3e0ee0a346be4bce14baec679ab11da17f679a9` |
| `csrc/lm/trace_lm_infer.c` | `d21d93aec29300d5dc42f545c05ab59dd0b235773fe329b17ee3ef24589fed6e` |
| `csrc/lm/trace_lm_kv.c` | `ae7bb258cb82ad10e013eb058bdff83592579c7be694ee8d642b1bd4df828354` |
| `csrc/lm/trace_lm_kv.h` | `37d1a1794cb08f702c62dd93c14bf5aa6100338a76606da0a71538c3c94cb57c` |
| `csrc/lm/trace_lm_loader.c` | `550ae9f15884dde3185ff4f539b420c874c49576588f0d5c52b7d5b97b4d82bb` |
| `csrc/lm/trace_tokenizer.c` | `c31ea099d5ee51efd57a0b7a6243ec0a60ff1795a1d46dd15a71a3688456fd81` |
| `csrc/lm/trace_tokenizer.h` | `95ea13066891216d26838555f171b52a768e46a5e78e265c97da6c3cc57c2606` |
| `csrc/lm/trace_grammar.c` | `e9f344d6c2f8e7ec499f82aea71935a031a3df4bc6ceac4445471703155cbec7` |
| `csrc/lm/trace_grammar.h` | `6d2cac4360e26ed348c9b87fcecceb84105448e6fcc600791055bf42de0a3f51` |
| `tools/import_bitnet.py` | `2695e1594aacd9afa43043b520b111002d5b52676190f03bce40a21da8b7ccdb` |
| `scripts/benchmark_bitnet_2b.py` | `ac13ccd811972c9c2748ef0464782c587c213f33270a16c77acb17e3a0915524` |
| `scripts/run_bench_4b.py` | `77e33d9eb3a3c566b061cb5c7738652128cd62733b59b49c6fca05ce3dad3909` |
| `results/eval_confirmatorio_5semillas_results.json` | `bf7c8890a1b46dfa7db89d3a6647e1cd64c0f1594d540569f058dad3c1dcc446` |
| `results/e0_higiene_results.json` | `91c1fc2c55bff3f51dd6632db27f8e537eb9691c8376a5b2d23b42d87b0709dc` |
| `results/e1_confirmatorio_5semillas_results.json` | `4729c36ee1a0b2bd2505c7d3cf66409b75eb97b63f90291a63d5e7431f9d9d36` |
| `results/e2_aislamiento_causal_results.json` | `31c8cdbac8ab62ccad849fe2f27d03a6bd0a315792e2ad1e7cd3295e412ae587` |
| `results/e4_caracterizacion_nvme_results.json` | `cc473e1a22671efe7768ced1d4357ddf40a400bc2a60b399c81c77e94efd87dc` |
| `results/eval_fase2b_unblock_5semillas_results.json` | `3115f8581a136d79d6cbf9f4b1eb974e28a073ebd11b58e18a59d06ef0b458e8` |
| `results/eval_fase1_bitnet_5semillas_results.json` | `e7eecd08940cb36343a16609977d135c1845666d3ffd6cba1f7c4b5ac6b45508` |
| `results/eval_fase1_5semillas_results.json` | `b64b18273a46b5328f4592b6ce21caab180b757263009d3d7ad64c718fa1c5ad` |
| `results/fase3_benchmark_results.json` | `15378aabf51244fa6b51c1584560993f84734cab2fc5f8088746f801c6f4852f` |
| `results/fase4b_bench_2b_c1_c3.json` | `c27b3b6ece953d36ac8bec36941256e28b67579c4595410be95c05548e481b78` |
| `trace/__init__.py` | `dae6ed04475a74aff183656512b99dcd6d7eee06eda423f150df651ea0dcd61f` |
| `trace/core/context_manager.py` | `c7c78a8964fba1af150f18e179921a664466474317f87f16a464f127e58300eb` |
| `trace/layers/abacus.py` | `c5678828d2441202a55483f983477a87c510fb30e5104ce918ec9dff74efa6b0` |
| `trace/layers/bitlinear.py` | `49145ddb27f4c359b40c601471979903db98818641551da47583bfc854fc185f` |
| `trace/layers/norm.py` | `e4bf2a5bed871a3b3fc407a213f4d77b5855e19ba4f67181768190244fa854af` |
| `trace/models/causal.py` | `15dbdbc06b5c598382c81006d8e645455d3f3a08e9cecef3e94685eeb2bc7136` |
| `trace/modules/attention.py` | `ef7fa463a5effa6d48f153c60337616c45dc74ef76363a21156dd1981366c028` |
| `trace/modules/paged_moe.py` | `ee084cdf8ccfc9e47b69a0aada76a2c1b55df93c1a310ac8e17627a54916181e` |
| `trace/modules/resident_core.py` | `55a9a27e3722eed1a1ce68c37d968db462e1f16174b2bdbcdf6174c8f8dad2b7` |
| `trace/ops/paged_kernel.py` | `2e153c380104c1dc4ea34b75c16a15e86de1c36454ee47db466dc65dd1c3eb0f` |
| `trace/ops/ternary_cpu.py` | `bfc9355bdbf60ddd77ed167fa1a94603a06ed9864a63840ad74a5a739168ac88` |
| `trace/serialization/paged_bitcos.py` | `23eba9900f495272629b01e7214c785f0d17ba5faf7e98b39dea89b906c0affe` |
| `trace/lm/__init__.py` | `a070ca52bb9c7de6741d35ca6022dab079632650680857b2494e637d215547e7` |
| `trace/lm/config.py` | `1937304ad72409a9572ecee19e5f7135c031b441a41b1f40dc7edb54bd91f245` |
| `trace/lm/gqa.py` | `e89f6f016a45dc8e1ad414dad43b14fe8a4555ce81b4cb3379bf15320ed110d8` |
| `trace/lm/mla.py` | `0aaf9dabd3fb5288bfd8fdac9de9658f65b70438ab72be8293fb69112f47e4ed` |
| `trace/lm/model.py` | `22b8b0aacb1acbc5a57b7cce71ca632d44069cd1fa33a81f3c0d251324e0c8a0` |
| `trace/lm/tokenizer.py` | `32d3caa696ca3bbc56abec457ef7d47b213be7d5f584140253045404c543dd16` |
| `trace/tir/__init__.py` | `3c2df6405a2128c6a4e8468af8cdb052b24a161ab6706738598785f126932093` |
| `trace/tir/sandbox.py` | `b0b76a847cfcfbfed6d26edbc5069cc861afffccb5ea389dc6e95e4a850ca37b` |
| `trace/tir/l3_store.py` | `6cd54ff10ee5ef8fdc42c662b66ccdb9b6f6aa15c2b011c47f47e8c2813b3370` |
| `trace/tir/tools.py` | `969b28ff4e082b8e1972df899c7f52b0303a3cf3b7a7ece7d61bedee5c0c68ed` |
| `trace/eval/__init__.py` | `4ded120729b842fe1658914116db7640993b30073061b1a536d10ef0d568bf0a` |
| `trace/eval/bench_4domains.py` | `b282265881c2f53a7aa4241f6c969302596acf13dce6e6de6670bb32037b3494` |
| `tests/test_paged_bitcos.py` | `38ac00c5193a5f0b63e27ac16472e062f39c73705885455ce8fddcc63cb09e41` |
| `tests/test_avx2_kernel.py` | `5801c8ef6490d6244ee307eeee3d65590b33517f079b531c2a04f790fae2d4dd` |
| `tests/test_parity_and_falsification.py` | `212ba6d1897a5a37adbaa859f34f4fa785f52abaddcbeb459621e5df36047cbd` |
| `tests/test_trace_infer.py` | `f8f5f6c719b3813e8f5bf7ba9f0e876b63084c24aa9ab907a64158c6cc15be2f` |
| `tests/test_smoke.py` | `38fb7fa976097e2f5448c483a7a26319f5ca5e4b0a46db9aec050d50cf54e4aa` |
| `tests/test_e0_hygiene.py` | `5cf5142a5a7d5ea3828ee476a6f9df8df8c9ad0a46d48ceccba917240c9eb5b2` |
| `tests/test_architecture_integrity.py` | `178c2a2fe0b1462a866e4c52339bee680d797fa5da93e1ee13015257268e42d9` |
| `tests/test_gqa_compact_emb.py` | `cc168c08d29461c76a4baddc7b8de4c13f8af7de8ea887a9f76a367f987cc804` |
| `tests/test_bitnet_importer.py` | `785840f11f68140ebee0a61979a113edba735adf2bc899d04663c6d2c0f0ca8b` |
| `tests/test_bitnet_ppl_harness.py` | `507258fc59dee7543a9c5b9290134a14cb74998b09244003d7ac8f3a15f4ab4f` |
| `tests/test_tir_sandbox.py` | `d2ae8048d16c61e1674d267819d378c457b3047dae1e6a22184d1dd8750a6566` |
| `tests/test_grammar_mask.py` | `1ff105b1c77d98ffb5732d53ad0818d00b6775fa45d476eacecce878edecf077` |
| `tests/test_bench_4domains.py` | `3eac2f710e6d44efface7ee9ffeee38706517000bd0aff71455c30fd622976a1` |
| `docs/FASE3_BENCHMARK.md` | `cb8265996d67ae5277790be9ec0c511c0c8cee74f4935cdba6441e897f9d61a9` |
| `docs/HITO_3_7_BENCHMARKS.md` | `5596dd19c391ab6c08cafeda448a4280b93e28149f8c902f3d11142c917e038d` |
| `docs/FASE4_BENCHMARK.md` | `6904ff69e013e5048bc38d0f50c70b00b20f55ba76d16fb0d19170b5ac377617` |
| `docs/PAPER_TECNICO_BASIC_LITE.md` | `8eda057ccbbe03bbb668080cf2f4092cfdccde967167f47a2c10ea3d18906e1c` |
| `docs/PROJECT_IDENTITY.md` | `a63ec504fac968e29faf6d0390282253921efc1f1441fb7501070dc1be0a07ac` |
| `docs/AI_ASSISTANCE.md` | `f17475ef92cb4e563f53afb5067705112a535abe1daafe4708b897e88f505db8` |
| `docs/PROVENANCE.md` | `7a418f2a1fac52ded0b8c0ad8a5057a57ae99999d8e724d64a6a626af4824ea7` |

---

## 3. Hashes Criptográficos Históricos de Referencia (Etiqueta congelada `phase2-v6-stable`: `c874140` / `113ee26`)

| Archivo Fuente Histórico | Hash Criptográfico SHA-256 |
| :--- | :--- |
| `train_v6.py` | `5e1ee09f7fd62994834de4c0a35473e7e6a08ac72f07397683480879149a5ee0` |
| `eval_v6.py` | `943f57638174a12c1ccddec07d9deeedea82a68bdf41ad2dad9234577647676e` |
| `eval_composition_audit.py` | `6e13bfa2aaca9ccdd82d7101469ff916520454c9d370bf8fdaa6e0402828a18e` |
| `data_v2.py` | `2b2acd2f2118a280344bc6627bb002f0e0956551f1b5d0787da970f327a45a41` |
| `trdm/models/trdm_v6_causal.py` | `1a8686cff07075ff143c4c5f584ebdefd148560b4ce2201c454d9f32009f1e28` |
| `v6_confirmatorio_5semillas_results.json` | `bf7c8890a1b46dfa7db89d3a6647e1cd64c0f1594d540569f058dad3c1dcc446` |

---

## 4. Pruebas Realizadas y Resultados Vinculantes Asociados

1. **Higiene de Medición (E0):** Verificación de causalidad temporal, aislamiento de muestras del lote y consistencia de parsers numéricos (`results/e0_higiene_results.json`).
2. **Evaluación Multi-Semilla Fase 1 (E1):** Evaluación pre-registrada con 5 semillas fijadas (`42, 123, 456, 789, 999`) reportada en `results/e1_confirmatorio_5semillas_results.json`.
3. **Aislamiento Causal (E2):** Matriz 2x2 para desacoplar decodificación iterativa frente a capacidad representacional (`results/e2_aislamiento_causal_results.json`).
4. **Caracterización Física NVMe (E4):** Caracterización de latencia y throughput de E/S en disco SSD NVMe (`results/e4_caracterizacion_nvme_results.json`).
5. **Evaluación Confirmatoria Fase 2 (TRDM-v6):** 3000 pasos en 5 semillas sobre 480 casos composicionales (`results/eval_confirmatorio_5semillas_results.json`).
6. **Evaluación Consolidada Fase 1 (TRACE v1.2.0):** 6000 pasos en 5 semillas bajo schedule BitNet b1.58 (`results/eval_fase1_bitnet_5semillas_results.json`, Full-EM $90.29\% \pm 13.44\%$, Scratchpad EM $100.00\%$).
7. **Evaluación Fase 3 (`45m-bi`, TRACE v1.4.0):** 47.2M bilingüe W1.58, `214–237 tok/s` en `36.22 MB` RAM (`docs/HITO_3_7_BENCHMARKS.md`, `results/fase3_benchmark_results.json`).
8. **Evaluación Fase 4A + 4B (Escala 2B + Banco 4 Dominios, TRACE v1.5.0):** Motor C AVX-512 VNNI sobre `BitNet-b1.58-2B-4T` (`pp512` `83.12 tok/s`, WikiText-2 PPL `16.24`, `1,255.3 MB` RAM, `docs/FASE4_BENCHMARK.md`) y matriz simétrica `C1–C3` (`results/fase4b_bench_2b_c1_c3.json`, `C1` `25.0%`, `C2` `50.0%` con `factual` `100%`, `C3` `37.5%`, `out_of_sandbox_executions == 0`, `285/285` tests en verde).

---

## 5. Instrucción de Verificación de Integridad

Para verificar la integridad criptográfica de cualquier archivo en una auditoría o peritaje técnico:
```bash
sha256sum <ruta_del_archivo>
```
La salida debe coincidir unívocamente con el valor registrado en la tabla anterior.
