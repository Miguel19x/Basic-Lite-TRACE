# Basic-Lite TRACE — Manifiesto de Versión y Criptografía (RELEASE_MANIFEST)

**Proyecto:** Basic-Lite TRACE  
**Versión Documentada:** TRACE 1.3.0+ (Motor Nativo C Modular `csrc/`, Fase 3 MLA/LM, Blindaje Legal y Encabezados SPDX)  
**Fecha de corte:** 24 de noviembre de 2026  
**Titular y Custodio:** Miguel19x (`miguelaranguren19x@outlook.com`)  

> [!NOTE]
> Este manifiesto sirve para identificar fehacientemente la composición e integridad de los archivos del paquete en la fecha especificada mediante huellas criptográficas SHA-256. No constituye por sí solo una certificación legal o notarial de prioridad, sino una prueba técnica inmutable de existencia previa e integridad (*prior art / proof of non-alteration*).

---

## 1. Entorno Físico y de Ejecución Documentado

* **Procesador (CPU):** Intel Core i3-1115G4 @ 3.00 GHz base (2 núcleos físicos / 4 hilos lógicos, 6 MB SmartCache L3).
* **Memoria RAM:** 16 GB DDR4 @ 3200 MT/s.
* **Almacenamiento secundario:** SSD NVMe PCIe Gen4 x4 (MSI M450 500GB).
* **Sistema Operativo:** Linux x86_64 (Kernel 6.x).
* **Herramientas de compilación y runtime:** GCC 13+ con soporte OpenMP y AVX2, Python 3.11+, PyTorch 2.x (CPU-only build).

---

## 2. Hashes Criptográficos SHA-256 de Archivos Fuente y Artefactos Clave (Versión Consolidada v1.3.0)

Los siguientes hashes han sido generados mediante el algoritmo estándar SHA-256 sobre el contenido exacto de los archivos consolidados de la versión 1.3.0:

| Archivo Fuente / Artefacto | Hash Criptográfico SHA-256 |
| :--- | :--- |
| `README.md` | `415d729c9201115a65ce4e91fe0cdefc240f3c187878d9d3067875eab6adcbe2` |
| `README.es.md` | `48e61f94dcf6d542bf13fa2b9b2f29a88365535a2fe9e48e04df9e3f289fed4a` |
| `LICENSE.txt` | `933abb675562927cd166d6130ee20dbdb5545613c9122019305cb05f44613143` |
| `LICENSE-EN.md` | `06827a519c33975b2e2ebc9e7f20342b8ff42c878166947fdbb0977b23d3f241` |
| `CONTRIBUTING.md` | `ae4af9dadeec6125b5600cccc69888b6cc8dec7e569067d64f05a10819af153f` |
| `CODE_OF_CONDUCT.md` | `31a86da1c334e7c5c79503392d135456356c2ec181000f8466eca270db264f6d` |
| `SECURITY.md` | `1ba029741067a5c87758b8f16103f123362dffbb4d5d0af1a26f6d6b969a34a1` |
| `NOTICE.md` | `f95611c431e750afb376053709a3fb897e51b1ff04bd4004ef4906f20b754d36` |
| `AUTHORS.md` | `2c7a099203dfdc56a25bb1eea549bce129600039d53ce9ff52b3ee24dc39d84c` |
| `CHANGELOG.md` | `6884379222c5568930185aa53352a39fcb6e8f15852c6e08f47e123a8a9401eb` |
| `THIRD_PARTY_NOTICES.md` | `c626ea954c9da9151b35058632cf0d675798a8ce96d05984948874b3a686b43c` |
| `Makefile` | `dcfd6ac1dd388c06a83e56b5badb9369ec1a0a9b0ce691694bc2a7fb712865c1` |
| `config.py` | `2633f978f5e01bf2456f22ed6e741f1afb4394b602924da406cd5c10d1532925` |
| `train.py` | `5063458ebf853a4182692d927be9b845dcab35c2400ce4f2fb640e35def75946` |
| `eval.py` | `b8dc7b07303df47f64d3209f9d094a54e4807102560911651aba166b597f3cff` |
| `data.py` | `6e88a400f637c0120c258bbe4bf03b671d0dfab7bf6a064d522bbdf053db37d3` |
| `export_paged.py` | `a833f52adefbe59ea628b3f84bdaf7170949e3aa5a376f828ac8bb34bfe6dc9d` |
| `export_paged_lm.py` | `949b429c0d0d698becdc8e1a2bfd201d1df644c5c3b5eb0bc3a6a8dc8690968e` |
| `eval_fase1_5semillas.py` | `fcb980df3d3d0c6ebdb85c6a8af93d8e60001da07239bc63e27ef8aba272db18` |
| `csrc/paged_dispatch.c` | `6367cedbb3dd4b9ceb68067719b37cc2c3f71cf539f1b62271bb89972df7d803` |
| `csrc/paged_dispatch.h` | `dc4cd0a6ed0b51d03b2dbf4e5db68c3033c6e371938081515a5c4a1fa7d6ad00` |
| `csrc/paged_gemm_avx2.c` | `b8a265d505f07e543efb254de0b49900413d7b86b1f5f3d7e2921abb239a966f` |
| `csrc/paged_gemm_scalar.c` | `3090b32c0c736e9c5d19272d96a7a38df7d8323bc2281fe55bb3d8c2f36d83f7` |
| `csrc/paged_unpack.h` | `4a6ddef34f3f56e54e62ef1e7442b73539e2ebe622d1a79be30b2e6994933864` |
| `csrc/trace_causal.c` | `6507de7e8653733b660960598a902be6631c6cd9bd20a8e77a49f84c1bf7873a` |
| `csrc/trace_infer.h` | `59fed297ba743f665647ec23dc720035d5d31fab1f76f9119e2a8683c33fbb17` |
| `csrc/trace_loader.c` | `5b675ba3cbdd38c936d5f1b6448a90a036ba7b70b8c1958efd5d76d1da124c28` |
| `csrc/trace_math.h` | `3e84931f67aab132120a03f644155cd39967ee782f54576e1e29078566f39814` |
| `csrc/lm/trace_lm.h` | `5e44fce882964ef51e998705d91b2f2aa89b30485c6c5cce5d7bd8b07130e382` |
| `csrc/lm/trace_lm_cli.c` | `396a431aea7f1e18265206a73a3424af1e7b07fdb3b6677783f54d78860b21b8` |
| `csrc/lm/trace_lm_infer.c` | `393b540621c42b9e1b87b1bff193759ca03842e7658ed82daea61501148e0883` |
| `csrc/lm/trace_lm_kv.c` | `15d7df01e5ce09f7a3e1046472d6cd4493681365d41e974304cded78751fdc5c` |
| `csrc/lm/trace_lm_kv.h` | `c37eea4d46cea44acf0cd951cfa2d943310c8592ac9ca31f3ee7f3e861e061b4` |
| `csrc/lm/trace_lm_loader.c` | `79f2213d3ae2324f5129e3fe92f2144c2b8d15e82fc1084d7cd91814e6a52046` |
| `csrc/lm/trace_tokenizer.c` | `b08cc0998d0a08c67f03a38b7d52f46d5c619aeb072d05c8a45409c07fb16ab0` |
| `csrc/lm/trace_tokenizer.h` | `7318790c01cb17c4d3900711e8e841ec6ab9d6e2f78f23aef5bb62d8e3cbb429` |
| `results/eval_confirmatorio_5semillas_results.json` | `bf7c8890a1b46dfa7db89d3a6647e1cd64c0f1594d540569f058dad3c1dcc446` |
| `results/e0_higiene_results.json` | `91c1fc2c55bff3f51dd6632db27f8e537eb9691c8376a5b2d23b42d87b0709dc` |
| `results/e1_confirmatorio_5semillas_results.json` | `4729c36ee1a0b2bd2505c7d3cf66409b75eb97b63f90291a63d5e7431f9d9d36` |
| `results/e2_aislamiento_causal_results.json` | `31c8cdbac8ab62ccad849fe2f27d03a6bd0a315792e2ad1e7cd3295e412ae587` |
| `results/e4_caracterizacion_nvme_results.json` | `cc473e1a22671efe7768ced1d4357ddf40a400bc2a60b399c81c77e94efd87dc` |
| `results/eval_fase2b_unblock_5semillas_results.json` | `3115f8581a136d79d6cbf9f4b1eb974e28a073ebd11b58e18a59d06ef0b458e8` |
| `results/eval_fase1_bitnet_5semillas_results.json` | `e7eecd08940cb36343a16609977d135c1845666d3ffd6cba1f7c4b5ac6b45508` |
| `results/eval_fase1_5semillas_results.json` | `b64b18273a46b5328f4592b6ce21caab180b757263009d3d7ad64c718fa1c5ad` |
| `results/fase3_benchmark_results.json` | `15378aabf51244fa6b51c1584560993f84734cab2fc5f8088746f801c6f4852f` |
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
| `trace/serialization/paged_bitcos.py` | `a4f822a18fce09699320a4a3fa1e0e9776c26f984c958b0365145aa124b1ef86` |
| `trace/lm/__init__.py` | `a070ca52bb9c7de6741d35ca6022dab079632650680857b2494e637d215547e7` |
| `trace/lm/config.py` | `2c072e01f7561ae99f6f7816a0175f57c1916bf31478d8712cf079b87aaf9a0c` |
| `trace/lm/mla.py` | `0aaf9dabd3fb5288bfd8fdac9de9658f65b70438ab72be8293fb69112f47e4ed` |
| `trace/lm/model.py` | `22b8b0aacb1acbc5a57b7cce71ca632d44069cd1fa33a81f3c0d251324e0c8a0` |
| `trace/lm/tokenizer.py` | `32d3caa696ca3bbc56abec457ef7d47b213be7d5f584140253045404c543dd16` |
| `tests/test_paged_bitcos.py` | `d71579315f1bf32558a40950d99c1914227cf5a5644335c28e5f2845fb9a27cb` |
| `tests/test_avx2_kernel.py` | `b40a62d1f8323d72686d09fb253713df780eaec25b08b68a4a3805bae3975161` |
| `tests/test_parity_and_falsification.py` | `c0c51ba569c451fa9f2a4e8086ff69e843f7988b2df49c91e1442c7dcd755658` |
| `tests/test_trace_infer.py` | `6c4be8bb8e685484bd0f7337c437128a83c0af00c8fc2d0562cc36947fd24fe9` |
| `tests/test_benchmark_fase2.py` | `7d0a10e23110a610e711e307e450766407a45f291bb267220e101b8018d03112` |
| `tests/test_smoke.py` | `d12de343561a43e87a90c38f89cba626da6fd2bb9b7dda751a40d04bfe8717c2` |
| `tests/test_e0_hygiene.py` | `23bad08aede6ac431091854f5875a01b102e1c228fae7d8e696950ba6b31afbc` |
| `tests/test_architecture_integrity.py` | `bda1692dbc60c1c245eafda607d129d052ae4f8c212af71a4513982a9dfa1fce` |
| `docs/PAPER_TECNICO_BASIC_LITE.md` | `91828322b27792d7c84fefb115bfae67a08136cb0a0558c1fdc9a03bb0f6dace` |
| `docs/PROJECT_IDENTITY.md` | `4cc4138fabd8b8dc9ee85d8006f93b5a0c1ee538500d99ca1626161b79b65bcd` |
| `docs/AI_ASSISTANCE.md` | `f17475ef92cb4e563f53afb5067705112a535abe1daafe4708b897e88f505db8` |
| `docs/PROVENANCE.md` | `7a418f2a1fac52ded0b8c0ad8a5057a57ae99999d8e724d64a6a626af4824ea7` |
| `docs/ROADMAP_TRACE.md` | `6de7ea4061c108b176fe0dd819e7763c4ce319fd4efc66711219d6a151fb4e93` |
| `docs/ROADMAP_POST_FASE5.md` | `045ec343aa1fae7e980a21d8a58f78988a2f60d3e4e9685ebea47511fa5daa2c` |

---

## 3. Hashes Criptográficos Históricos de Referencia (Etiqueta congelada `phase2-v6-stable`: `c874140` / `113ee26`)

| Archivo Fuente Histórico | Hash Criptográfico SHA-256 |
| :--- | :--- |
| `train_v6.py` | `5e1ee09f7fd62994834de4c0a35473e7e6a08ac72f07397683480879149a5ee0` |
| `eval_v6.py` | `943f57638174a12c1ccddec07d9deeedea82a68bdf41ad2dad9234577647676e` |
| `eval_composition_audit.py` | `6e13bfa2aaca9ccdd82d7101469ff916520454c9d370bf8fdaa6e0402828a18e` |
| `data_v2.py` | `2b2acd2f2118a280344bc6627bb002f0e0956551f1b5d0787da970f327a45a41` |
| `trdm/models/trdm_v6_causal.py` | `1a8686cff07075ff143c4c5f584ebdefd148560b4ce2201c454d9f32009f1e28` |
| `Protocolo pre-registrado — Basic-Lite (TRDM-v5), Fase 1.md` | `60668166afc9c904e25553ba29c3f8b91892153da301069e76ab32e485cd0f39` |
| `v6_confirmatorio_5semillas_results.json` | `bf7c8890a1b46dfa7db89d3a6647e1cd64c0f1594d540569f058dad3c1dcc446` |

---

## 4. Pruebas Realizadas y Resultados Vinculantes Asociados

1. **Higiene de Medición (E0):** Verificación de causalidad temporal, aislamiento de muestras del lote y consistencia de parsers numéricos (`e0_higiene_results.json`).
2. **Evaluación Multi-Semilla Fase 1 (E1):** Evaluación pre-registrada con 5 semillas fijadas (`42, 123, 456, 789, 999`) reportada en `e1_confirmatorio_5semillas_results.json`.
3. **Aislamiento Causal (E2):** Matriz 2x2 para desacoplar decodificación iterativa frente a capacidad representacional (`e2_aislamiento_causal_results.json`).
4. **Caracterización Física NVMe (E4):** Caracterización de latencia y throughput de E/S en disco SSD NVMe (`e4_caracterizacion_nvme_results.json`).
5. **Evaluación Confirmatoria Fase 2 (TRDM-v6):** 3000 pasos en 5 semillas sobre 480 casos composicionales reportados en `eval_confirmatorio_5semillas_results.json` (etiqueta histórica `v6_confirmatorio_5semillas_results.json`, Full-EM $5.12\% \pm 4.88\%$, Scratchpad EM $81.88\% \pm 10.14\%$, Validez sintáctica $100\%$).
6. **Evaluación Exploratoria Fase 2b (TRACE v1.1.0):** 3000 pasos en 5 semillas sobre 480 casos composicionales reportados en `results/eval_fase2b_unblock_5semillas_results.json` (Full-EM $19.04\% \pm 22.73\%$, pico $58.13\%$, Scratchpad EM $98.21\% \pm 1.01\%$, veredicto H-A1 Inconcluso).
7. **Evaluación Consolidada Fase 1 (TRACE v1.2.0):** 6000 pasos en 5 semillas bajo schedule BitNet b1.58 de 2 etapas reportados en `results/eval_fase1_bitnet_5semillas_results.json` (Full-EM $90.29\% \pm 13.44\%$, pico $98.96\%$ en Semilla 456, Scratchpad EM $100.00\% \pm 0.00\%$, ECE $0.61\%$, veredicto Puerta de Salida Aprobado).

---

## 5. Instrucción de Verificación de Integridad

Para verificar la integridad criptográfica de cualquier archivo en una auditoría o peritaje técnico:
```bash
sha256sum <ruta_del_archivo>
```
La salida debe coincidir unívocamente con el valor registrado en la tabla anterior.
