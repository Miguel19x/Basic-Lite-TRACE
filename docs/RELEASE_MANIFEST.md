# Basic-Lite TRACE — Manifiesto de Versión y Criptografía (RELEASE_MANIFEST)

**Proyecto:** Basic-Lite TRACE  
**Versión Documentada:** TRACE 1.1.0 (Desbloqueo Aritmético y Causalidad Decimal)  
**Fecha de corte:** 21 de noviembre de 2026  
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

## 2. Hashes Criptográficos SHA-256 de Archivos Fuente y Artefactos Clave (Versión Consolidada v1.1.0)

Los siguientes hashes han sido generados mediante el algoritmo estándar SHA-256 sobre el contenido exacto de los archivos consolidados de la versión 1.1.0:

| Archivo Fuente / Artefacto | Hash Criptográfico SHA-256 |
| :--- | :--- |
| `README.md` | `8ad333a07e41de7d3178f6cbcecf560cbf0c72f9ce4f775e97cdfdf3cc05869b` |
| `LICENSE.txt` | `7c4d8c21fddee9f087e23349cbeec3d8d513f37aaf54957f5c21ab8dfb3d2609` |
| `CONTRIBUTING.md` | `ed625c617b50377005202033c3b2d96a1c9e37b886904b029f340ffb206a8ff2` |
| `NOTICE.md` | `cff498beb420200e7237b98792229f9acdf3b0c5ebe935c7cad51bba5836351b` |
| `AUTHORS.md` | `2c7a099203dfdc56a25bb1eea549bce129600039d53ce9ff52b3ee24dc39d84c` |
| `CHANGELOG.md` | `0de4ce76b90b964b562422a59e670f7366b96ff57f9845e6589b501093c38da1` |
| `THIRD_PARTY_NOTICES.md` | `0aa1c14c11e32027a936a366616dafea6139ef02a619d19c861503784fc238c9` |
| `config.py` | `8fa667b11da6f570e35b3ea6fd965cbdb4882edf4258dce2bfb7ee615b75c3be` |
| `train.py` | `3efe5148fdafc9f07815bddbc1dd7c9c5a451cdf5ab82a067f7b9de181afe627` |
| `eval.py` | `36dabb3b8b7f303b2df3250046e842fe779049ad45c6193b92a40e6e8d50ce35` |
| `data.py` | `378c9178d28c9132683da925e7e97eab2376565698e57f159cfdea03a34675ef` |
| `paged_gemm_kernel.c` | `ff4fac91f34428c0bb1c90c99a85b99b19a9792814385a013cba15bc1348d228` |
| `results/eval_confirmatorio_5semillas_results.json` | `bf7c8890a1b46dfa7db89d3a6647e1cd64c0f1594d540569f058dad3c1dcc446` |
| `results/e0_higiene_results.json` | `91c1fc2c55bff3f51dd6632db27f8e537eb9691c8376a5b2d23b42d87b0709dc` |
| `results/e1_confirmatorio_5semillas_results.json` | `4729c36ee1a0b2bd2505c7d3cf66409b75eb97b63f90291a63d5e7431f9d9d36` |
| `results/e2_aislamiento_causal_results.json` | `31c8cdbac8ab62ccad849fe2f27d03a6bd0a315792e2ad1e7cd3295e412ae587` |
| `results/e4_caracterizacion_nvme_results.json` | `cc473e1a22671efe7768ced1d4357ddf40a400bc2a60b399c81c77e94efd87dc` |
| `results/eval_fase2b_unblock_5semillas_results.json` | `3115f8581a136d79d6cbf9f4b1eb974e28a073ebd11b58e18a59d06ef0b458e8` |
| `trace/__init__.py` | `cfb0f0af55c88e224a5e9850a6431c13de824a0b3332bbb461e8fc401497aa6c` |
| `trace/core/context_manager.py` | `96e134bc6bb5a7e851e7068c92aea0636b15ed318432ce4a21892fed3d214287` |
| `trace/core/weight_manager.py` | `500208a0429122dddbdb9d7f58df23af1ceec537c63a5621ecfa8be0794c9bac` |
| `trace/layers/abacus.py` | `f8c12469b63ac07849a4253a6c3a16e4775855733bf10d2d37725b6ced3b6baf` |
| `trace/layers/bitlinear.py` | `cd783d92bf48beae204fcbe53ffe5beecc1ef8d54c91d06391b27fa6565c1e4b` |
| `trace/layers/byte_patcher.py` | `ae00d6e40bc81358c21e948b73d48df385f612857dbbf778796564afe8f2f1f1` |
| `trace/models/causal.py` | `c75ffce1faf87218a0cc92e2f6cf43f069f818479f33294df9b70eec6968dd32` |
| `trace/modules/attention.py` | `66f40651d25a530f721f1a8889bfc61f4768ff9cedff5ae419eb58aab9492ac8` |
| `trace/modules/paged_moe.py` | `b61af0d57408a621b9e264c2e03d6fdd2899d3a6080d9add852476d42aa343ef` |
| `trace/modules/mla_attention.py` | `1520ee69177ff27a1889bbc47b3d15b854e4554be6f7a440dbcd72b3a9defb6d` |
| `trace/modules/resident_core.py` | `b81bbd22fa6819ae9f5e08f022b8ae464160c58c5e9ced9688d73e596ac94791` |
| `trace/serialization/paged_bitcos.py` | `f873cdc079cd478e3dd7022f7ce93175069a237ab5418b0c08d59eb873f5cdda` |
| `docs/PAPER_TECNICO_BASIC_LITE.md` | `e07e22f996979bfd8dc252507676e74656c7d4f5e93a1718c1711100bb29b1de` |
| `docs/PROJECT_IDENTITY.md` | `4cc4138fabd8b8dc9ee85d8006f93b5a0c1ee538500d99ca1626161b79b65bcd` |
| `docs/AI_ASSISTANCE.md` | `f17475ef92cb4e563f53afb5067705112a535abe1daafe4708b897e88f505db8` |
| `docs/PROVENANCE.md` | `fc563b48a81669ae3ba86f8b26d1009877478b58055fcf50184712d5ad12a766` |
| `docs/ROADMAP_TRACE.md` | `19d0578458362880f5e3a83f09eeecddacab542c2910876354dad8f39c42ade5` |

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

## 3. Pruebas Realizadas y Resultados Vinculantes Asociados

1. **Higiene de Medición (E0):** Verificación de causalidad temporal, aislamiento de muestras del lote y consistencia de parsers numéricos (`e0_higiene_results.json`).
2. **Evaluación Multi-Semilla Fase 1 (E1):** Evaluación pre-registrada con 5 semillas fijadas (`42, 123, 456, 789, 999`) reportada en `e1_confirmatorio_5semillas_results.json`.
3. **Aislamiento Causal (E2):** Matriz 2x2 para desacoplar decodificación iterativa frente a capacidad representacional (`e2_aislamiento_causal_results.json`).
4. **Caracterización Física NVMe (E4):** Caracterización de latencia y throughput de E/S en disco SSD NVMe (`e4_caracterizacion_nvme_results.json`).
5. **Evaluación Confirmatoria Fase 2 (TRDM-v6):** 3000 pasos en 5 semillas sobre 480 casos composicionales reportados en `eval_confirmatorio_5semillas_results.json` (etiqueta histórica `v6_confirmatorio_5semillas_results.json`, Full-EM $5.12\% \pm 4.88\%$, Scratchpad EM $81.88\% \pm 10.14\%$, Validez sintáctica $100\%$).
6. **Evaluación Exploratoria Fase 2b (TRACE v1.1.0):** 3000 pasos en 5 semillas sobre 480 casos composicionales reportados en `results/eval_fase2b_unblock_5semillas_results.json` (Full-EM $19.04\% \pm 22.73\%$, pico $58.13\%$, Scratchpad EM $98.21\% \pm 1.01\%$, veredicto H-A1 Inconcluso).

---

## 4. Instrucción de Verificación de Integridad

Para verificar la integridad criptográfica de cualquier archivo en una auditoría o peritaje técnico:
```bash
sha256sum <ruta_del_archivo>
```
La salida debe coincidir unívocamente con el valor registrado en la tabla anterior.
