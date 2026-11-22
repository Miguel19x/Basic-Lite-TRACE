# Basic-Lite TRACE — Manifiesto de Versión y Criptografía (RELEASE_MANIFEST)

**Proyecto:** Basic-Lite TRACE  
**Versión Documentada:** TRACE 1.3.0 (Motor Autónomo C, Paged BitCOS v1, AVX2 Fisionado y Cierre de Fase 2)  
**Fecha de corte:** 22 de noviembre de 2026  
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
| `README.md` | `383f2542a8d977966db8dcbff0adf1372bf20f0ad4873c1c522ec93132707105` |
| `LICENSE.txt` | `7c4d8c21fddee9f087e23349cbeec3d8d513f37aaf54957f5c21ab8dfb3d2609` |
| `CONTRIBUTING.md` | `ed625c617b50377005202033c3b2d96a1c9e37b886904b029f340ffb206a8ff2` |
| `NOTICE.md` | `cff498beb420200e7237b98792229f9acdf3b0c5ebe935c7cad51bba5836351b` |
| `AUTHORS.md` | `2c7a099203dfdc56a25bb1eea549bce129600039d53ce9ff52b3ee24dc39d84c` |
| `CHANGELOG.md` | `27d9222c1723a60e8279e761cdc325f671fe216ef35b1c65baa3f877f39b1b17` |
| `THIRD_PARTY_NOTICES.md` | `0aa1c14c11e32027a936a366616dafea6139ef02a619d19c861503784fc238c9` |
| `Makefile` | `aeeba40778b361613402cd75a6b4ccb7d48bf08fdaf352b4f8ab0c76f601cfae` |
| `config.py` | `8fa667b11da6f570e35b3ea6fd965cbdb4882edf4258dce2bfb7ee615b75c3be` |
| `train.py` | `4533971960c5f7b9e76ff7c2a354466bdc2feb1df43163dbd885087ae9aebd2d` |
| `eval.py` | `db11f0ee132c9156489c34e84f864de6c58663c2ae20718605bc520346efdf03` |
| `data.py` | `378c9178d28c9132683da925e7e97eab2376565698e57f159cfdea03a34675ef` |
| `export_paged.py` | `5d8ed1cee48c32bafce39c59747b41cbb73c5b64d958a2cf24b3b8fc6cc5bbed` |
| `paged_gemm_kernel.c` | `f5e9569e32b9379486b40fff5424e0bf1d105f0e76e85f5586eb5c8fdb2187c9` |
| `trace_infer.c` | `0f62f8186399dde61f0cf238e1df5e588c19c205694526f9bcb899867a193ffe` |
| `trace_infer.h` | `d20c645d3bf6eb410d90c48b9601c35005f2e6733be4927455f0a7cf961bc9f4` |
| `results/eval_confirmatorio_5semillas_results.json` | `bf7c8890a1b46dfa7db89d3a6647e1cd64c0f1594d540569f058dad3c1dcc446` |
| `results/e0_higiene_results.json` | `91c1fc2c55bff3f51dd6632db27f8e537eb9691c8376a5b2d23b42d87b0709dc` |
| `results/e1_confirmatorio_5semillas_results.json` | `4729c36ee1a0b2bd2505c7d3cf66409b75eb97b63f90291a63d5e7431f9d9d36` |
| `results/e2_aislamiento_causal_results.json` | `31c8cdbac8ab62ccad849fe2f27d03a6bd0a315792e2ad1e7cd3295e412ae587` |
| `results/e4_caracterizacion_nvme_results.json` | `cc473e1a22671efe7768ced1d4357ddf40a400bc2a60b399c81c77e94efd87dc` |
| `results/eval_fase2b_unblock_5semillas_results.json` | `3115f8581a136d79d6cbf9f4b1eb974e28a073ebd11b58e18a59d06ef0b458e8` |
| `results/eval_fase1_bitnet_5semillas_results.json` | `e7eecd08940cb36343a16609977d135c1845666d3ffd6cba1f7c4b5ac6b45508` |
| `results/eval_fase1_5semillas_results.json` | `b64b18273a46b5328f4592b6ce21caab180b757263009d3d7ad64c718fa1c5ad` |
| `eval_fase1_5semillas.py` | `97c6be7747a2aefd6fd4ddcadbf39a219affad8e44414aa3d4ecd73b03e6027a` |
| `trace/__init__.py` | `cfb0f0af55c88e224a5e9850a6431c13de824a0b3332bbb461e8fc401497aa6c` |
| `trace/core/context_manager.py` | `96e134bc6bb5a7e851e7068c92aea0636b15ed318432ce4a21892fed3d214287` |
| `trace/core/weight_manager.py` | `500208a0429122dddbdb9d7f58df23af1ceec537c63a5621ecfa8be0794c9bac` |
| `trace/layers/abacus.py` | `f8c12469b63ac07849a4253a6c3a16e4775855733bf10d2d37725b6ced3b6baf` |
| `trace/layers/bitlinear.py` | `74ee510c04fb4c285bb5b184df597980f1af8dec3dc0bdcd2b1d3529a92d361a` |
| `trace/layers/byte_patcher.py` | `ae00d6e40bc81358c21e948b73d48df385f612857dbbf778796564afe8f2f1f1` |
| `trace/layers/norm.py` | `1aa6c7f3644172c5090ce172a67391e61d9690f7ff0d4d06e734abc62fbbea13` |
| `trace/models/causal.py` | `2607970610d5cfa23d7d4583a0976ab599eb73eb3cbff417522b5da78355cec4` |
| `trace/modules/attention.py` | `66f40651d25a530f721f1a8889bfc61f4768ff9cedff5ae419eb58aab9492ac8` |
| `trace/modules/paged_moe.py` | `b61af0d57408a621b9e264c2e03d6fdd2899d3a6080d9add852476d42aa343ef` |
| `trace/modules/mla_attention.py` | `1520ee69177ff27a1889bbc47b3d15b854e4554be6f7a440dbcd72b3a9defb6d` |
| `trace/modules/resident_core.py` | `b81bbd22fa6819ae9f5e08f022b8ae464160c58c5e9ced9688d73e596ac94791` |
| `trace/ops/paged_kernel.py` | `ec87f753749e2e1a9b20817313226a260139799bca2af2696e621b7d0430f8bb` |
| `trace/ops/ternary_cpu.py` | `5334a6c1f12a5a52db726242784d6e094ebcbb5d0bd4a8aafb51197fc5d0a2e4` |
| `trace/serialization/paged_bitcos.py` | `2709b0a29284a8ba79078a3b374ea4a82b086b2b760972d31e370b926af61f1a` |
| `tests/test_paged_bitcos.py` | `24eb4a148b91cf05ae9fdb7afb51fcb42d525cdc198a33f0180f5b4970f4b602` |
| `tests/test_avx2_kernel.py` | `85fac2c04ad1e01daafc12259b8471713b726bc7c86e927aadf7476dcaec9ec1` |
| `tests/test_parity_and_falsification.py` | `e122b20967e91ba9d4efbddec192222ed5918ce12131c426c36e65b70562a6ea` |
| `tests/test_trace_infer.py` | `3f291d4335fda3ef5cbff6d31459e54fc249b640cdef5c22a02ee6a12efd9920` |
| `tests/test_benchmark_fase2.py` | `a1863a61d98a3632c2b1c95626088c8484f27811100e328bf3c88d1d84d134e9` |
| `tests/test_smoke.py` | `cf0ac0e331bf71bbfbe712c5a04f558b6660bfc016adabc99eb7fd07938347e3` |
| `tests/test_e0_hygiene.py` | `19dd184b0290ef88773c3fc1a8bfb31bbb51ff25b65464ba3c819021f4142ecc` |
| `tests/test_scalability_flags.py` | `78fa980271e1c154bb947f82b702fda0d2c02b2032bbeca5be0d986c8c1fd9df` |
| `docs/PAPER_TECNICO_BASIC_LITE.md` | `e07e22f996979bfd8dc252507676e74656c7d4f5e93a1718c1711100bb29b1de` |
| `docs/PROJECT_IDENTITY.md` | `4cc4138fabd8b8dc9ee85d8006f93b5a0c1ee538500d99ca1626161b79b65bcd` |
| `docs/AI_ASSISTANCE.md` | `f17475ef92cb4e563f53afb5067705112a535abe1daafe4708b897e88f505db8` |
| `docs/PROVENANCE.md` | `fc563b48a81669ae3ba86f8b26d1009877478b58055fcf50184712d5ad12a766` |
| `docs/ROADMAP_TRACE.md` | `6de7ea4061c108b176fe0dd819e7763c4ce319fd4efc66711219d6a151fb4e93` |
| `docs/ROADMAP_POST_FASE5.md` | `61ca2b65c1ff1bf63310001ec572760dff4cdd99ac9a6e7681d4a3e5e7970c77` |

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
