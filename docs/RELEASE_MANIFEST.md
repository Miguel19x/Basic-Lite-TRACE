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
| `README.md` | `bf55dfcc547808fcd59f5dba077d0b799fdb867bff9006d7b379eaf4c5dd60f5` |
| `LICENSE.txt` | `933abb675562927cd166d6130ee20dbdb5545613c9122019305cb05f44613143` |
| `LICENSE-EN.md` | `06827a519c33975b2e2ebc9e7f20342b8ff42c878166947fdbb0977b23d3f241` |
| `CONTRIBUTING.md` | `ae4af9dadeec6125b5600cccc69888b6cc8dec7e569067d64f05a10819af153f` |
| `CODE_OF_CONDUCT.md` | `31a86da1c334e7c5c79503392d135456356c2ec181000f8466eca270db264f6d` |
| `SECURITY.md` | `1ba029741067a5c87758b8f16103f123362dffbb4d5d0af1a26f6d6b969a34a1` |
| `NOTICE.md` | `f95611c431e750afb376053709a3fb897e51b1ff04bd4004ef4906f20b754d36` |
| `AUTHORS.md` | `2c7a099203dfdc56a25bb1eea549bce129600039d53ce9ff52b3ee24dc39d84c` |
| `CHANGELOG.md` | `6884379222c5568930185aa53352a39fcb6e8f15852c6e08f47e123a8a9401eb` |
| `THIRD_PARTY_NOTICES.md` | `c626ea954c9da9151b35058632cf0d675798a8ce96d05984948874b3a686b43c` |
| `Makefile` | `6fb5ef8443e8b8d4435a122ab7ef4e011d2cf50c7e4994811d8fcd83f6ea693d` |
| `config.py` | `a92ffe3822bd613b8263faec7d3a5002611afe7576e38309cc639ecf0a5499ae` |
| `train.py` | `edf1aade1ca00b92a140a26c98653c739ac5607c7e70ec4a2e9450b3a4ca8198` |
| `eval.py` | `a360b9e75d25a241987fede319e71e8023f414601bbfb7d9da57ea600ba97464` |
| `data.py` | `dc8e555db969600c950b88497d69cc59cb187f601cc1a9eadca1110d8e26bf8c` |
| `export_paged.py` | `b8f7bbb256b60a3a650786cd7fddf21196a188b4465f149749657781a52b96be` |
| `export_paged_lm.py` | `045fd365f3ff521ea23f181932c2ff85f147a1ac5f0844adf7afaf45a1e0de04` |
| `eval_fase1_5semillas.py` | `8344640ce4a3081c82cf170f77dd28dd84c28b512271532b4b7457f9b5733cc4` |
| `csrc/paged_dispatch.c` | `32042af96e600d154209865b1ad410e436e03f794b96f5d8597a0c71b9a4f077` |
| `csrc/paged_dispatch.h` | `dc4cd0a6ed0b51d03b2dbf4e5db68c3033c6e371938081515a5c4a1fa7d6ad00` |
| `csrc/paged_gemm_avx2.c` | `49b542bd1db5c5e830dcd3284d3ba38a83806d13302e2562ccf680413820f586` |
| `csrc/paged_gemm_scalar.c` | `24f721558d253fdb45c2802cda80fcdf0e6144a5fc5a771ba4756936a55bd175` |
| `csrc/paged_unpack.h` | `4a6ddef34f3f56e54e62ef1e7442b73539e2ebe622d1a79be30b2e6994933864` |
| `csrc/trace_causal.c` | `0555944a527d5d4d6f33ed499129d71d6225573a57a7ea267d98a15eb93bd16c` |
| `csrc/trace_infer.h` | `59fed297ba743f665647ec23dc720035d5d31fab1f76f9119e2a8683c33fbb17` |
| `csrc/trace_loader.c` | `a12b47631a79177b44acf074658a99d2d99c093dc0d9059dc5c3dfeee3ff241e` |
| `csrc/trace_math.h` | `286043da9f8b19c42e00119ab60733a5874708be9098660aa8f648a7c303a356` |
| `csrc/lm/trace_lm.h` | `8a1cef07ab5cc87891544f2866a85aa92ce1946b50c9a527f10950bc7a132503` |
| `csrc/lm/trace_lm_cli.c` | `b1ea92344cffd4c6b7ac4210791f1a51b8ccf726f132b0bed983bac139552713` |
| `csrc/lm/trace_lm_infer.c` | `cd6bbac460dce19acba38bdc9e15ca47f42f80043fba5b2ab089278518464d35` |
| `csrc/lm/trace_lm_kv.c` | `f8a737925ac58f8d4c06d67b812c8897b9202dfd52d45786797dea22931877d1` |
| `csrc/lm/trace_lm_kv.h` | `c37eea4d46cea44acf0cd951cfa2d943310c8592ac9ca31f3ee7f3e861e061b4` |
| `csrc/lm/trace_lm_loader.c` | `f23d294a02fcbc7ec897eca14a0ac8877d3846440946853bf859827fbb9ad6fa` |
| `csrc/lm/trace_tokenizer.c` | `c7dcff829c4607a11caa23ae71a798c547be6b5972dc22c8e725b2ae86bfdfe3` |
| `csrc/lm/trace_tokenizer.h` | `1250c302585e5bc6f8a1f7fc629da1703f69d64e1f63c0a3b78323e2762b9942` |
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
| `trace/core/context_manager.py` | `31f549d40ea4f570fabb99048de606868163546de94b299f4634e6117106f9c6` |
| `trace/layers/abacus.py` | `c4532693f17e45c4ec9903b46269abc3ff682fb0bacabe31ce9e1ec90c35fa4f` |
| `trace/layers/bitlinear.py` | `47d083298a44d3afd3c4e784de9b65f097274155e927d89c9f39ad10cb312c41` |
| `trace/layers/norm.py` | `e4bf2a5bed871a3b3fc407a213f4d77b5855e19ba4f67181768190244fa854af` |
| `trace/models/causal.py` | `9616891093fa4eff7e2c3f3e38fc0e7d9da2597f7b702484efa7822f2b45036d` |
| `trace/modules/attention.py` | `41965c37e103f408fd50c951045afb282d33f2ff98cb751682ee647b19e5802e` |
| `trace/modules/paged_moe.py` | `3081e442f1c688e7671f90d4eee1150dec667a026cb8c5e8fc59a238b95bc188` |
| `trace/modules/resident_core.py` | `4b04f08493c1a5e0bb8cfce3a31fbf1b6e3b38f9a347ddcf6b76b36af08f4e2e` |
| `trace/ops/paged_kernel.py` | `186bcdf857f802f0b47fd7eb415ddab4e1b98782edec3df7457b4c708043660e` |
| `trace/ops/ternary_cpu.py` | `05e0c0c72853d9d0bab1c3054f53a06cf792d9e46e214fa0a40692df4f20c1d7` |
| `trace/serialization/paged_bitcos.py` | `8e0fac8e5e215cbc35816d03ad31bd7f7b4d5cd341c1c28304547b57d0a63a09` |
| `trace/lm/__init__.py` | `a070ca52bb9c7de6741d35ca6022dab079632650680857b2494e637d215547e7` |
| `trace/lm/config.py` | `2c072e01f7561ae99f6f7816a0175f57c1916bf31478d8712cf079b87aaf9a0c` |
| `trace/lm/mla.py` | `e7067e8562889eb3a0962bfc963ff895af5e06d92786b8e6c5c684b2235888c5` |
| `trace/lm/model.py` | `7df4eab05e2f78d8da749ddf71defc1152534415cec5c15c6ae2a9dbc219e20b` |
| `trace/lm/tokenizer.py` | `4ddb0ace530bd6c09afde2ebe274a021918f72930c00290c0fd8b1a7a2d657bd` |
| `tests/test_paged_bitcos.py` | `7c368f7ace7a265a21ddc8d910b69cfd3aaaa0d898de6b163c2f9225c9a9d356` |
| `tests/test_avx2_kernel.py` | `5aae790843c9721e338f51fc262fd904378d07dad571382e741daaa36d8e97da` |
| `tests/test_parity_and_falsification.py` | `550337223438e0ee404221a658a80581d2e97c58f4e425662a37886456766b56` |
| `tests/test_trace_infer.py` | `cf3efaab3bf7129d02d821c37d42af010cfcddfbb2e9a428b7befa384b8d8cd2` |
| `tests/test_benchmark_fase2.py` | `49d1cc3977682ca2917a33db15e256d8695dcbe31c692e4282de8b19166d9760` |
| `tests/test_smoke.py` | `d4cf16ef3a58ef244fd314390b836066db176e997258aeadc109ea7e62c7fd97` |
| `tests/test_e0_hygiene.py` | `57f693a15d1eed957e0f057d94519f32270838be9b66a0d55105ace4ca3ce520` |
| `tests/test_architecture_integrity.py` | `5a4c74eb171dea3bb52cbc260bb3fcbff8800eec082a202e02cbd5939c99e955` |
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
