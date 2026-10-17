# Allineamento del fork a FIDESlib 2.2.0

Base precedente del fork: `fabb8d2`. Upstream integrato: `597a0b3`
(`CAPS-UMU/FIDESlib/main`, FIDESlib 2.2.0). Al momento dell'integrazione
upstream non pubblica un tag `v2.2.0`: il riferimento riproducibile è il commit.
La cronologia del fork è conservata con un merge, senza riscriverla.

## Funzionalità upstream nuove rispetto alla nostra base

| Area | Funzionalità |
| --- | --- |
| Bootstrap | StC-first/decode-first: `EvalBootstrapStCFirst`, setup con `btsfirstboot`, variante iterativa; controlli `BootstrapMode` per evitare setup/evaluazioni incompatibili. |
| CKKS complesso | API `SetCKKSDataType(COMPLEX)`/`GetCKKSDataType`, encoding, aritmetica e bootstrap complessi. |
| Ritorno server → client | `UnloadCiphertext`, serializzazione/deserializzazione dei ciphertext, esportazione della base estesa QL·P senza chiave segreta. |
| API mutable | Implementazioni GPU delle varianti `Eval*Mutable*`, coerenti con OpenFHE. |
| Plaintext sparsi | Layout GPU compresso a `2*slots` valori per limb, usato direttamente dai kernel e dalle matrici del bootstrap. |
| Prestazioni | Lazy Shoup/reduction, NTT/INTT portabili, rescale con una moltiplicazione, weighted sum con meno temporanei, ottimizzazioni Chebyshev/Paterson–Stockmeyer. |
| Interoperabilità | Confronti bit-exact CPU/GPU anche per bootstrap sparse encapsulated, somme con offset, FIXEDAUTO/FIXEDMANUAL e base estesa. `SetLevel` elimina solo i tower. |
| Strumenti | Suite `OpenFheCompatTests`, coverage locale e riprendibile, report di uso kernel, timeline NVTX con consumo di memoria, selezione automatica delle architetture CUDA. |

Il dettaglio delle modifiche upstream è in [CHANGES.md](../CHANGES.md).
La dipendenza passa da OpenFHE 1.5.1 a **OpenFHE 1.6.1 patchato**; sono
conservati i tre modi upstream `patch`, `patch-minimal`, `port`. La verifica
bit-exact usa `patch`, con `WITH_NATIVEOPT=OFF` e librerie statiche.

## Sovrapposizioni e decisioni

| Sovrapposizione | Risoluzione |
| --- | --- |
| Bootstrap iterativo/Meta-BTS | Rimossa la nostra implementazione `IterativeBootstrap`; `EvalBootstrap` e `EvalBootstrapInPlace` usano gli overload upstream `Bootstrap(..., prescaled, numIterations, precision)`. Si adottano allineamento dei livelli, supporto delle tecniche di scaling e test esatti upstream. |
| Correzioni `addPt`/`subPt` e adattamento scale/livelli | Implementazione upstream, inclusi FIXEDAUTO e il segno della sottrazione; ritirati i confronti locali con tolleranza aumentata. |
| Riferimento al contesto in `LoadContext` | Upstream applica già la stessa correzione: il contesto viene costruito nella posizione definitiva prima delle chiavi. Si conservano anche le nostre impostazioni dei budget prima del caricamento. |
| Unload ciphertext upstream e nostro offload | Coesistono: `UnloadCiphertext` aggiorna l'oggetto CPU e rimuove l'handle GPU; `Offload` mantiene l'handle e uno snapshot per il ripristino automatico. Export e serializzazione usano il percorso upstream, anche dopo un offload. |
| Compressione plaintext e nostra cache bootstrap | Layout e aritmetica upstream. Adattati gli snapshot del fork: anche in RAM conservano i valori compressi, caricabili direttamente dalla memoria pinned. Contabilità VRAM basata sulle allocazioni effettive. |
| Sparse encapsulated e multi-GPU | Il nuovo key-switch GHS upstream è solo single-GPU. Lo si esegue su un contesto dedicato sul primo device, trasferendo i soli limb attivi in ingresso/uscita; il resto del bootstrap rimane distribuito. |
| Rescale di plaintext compressi | Corretto un refuso upstream: il secondo kernel NTT forward usava `INTT_grid_dim_X`. Per lunghezze con logaritmo dispari elaborava solo parte dei dati, causando NaN nelle regressioni del fork. Si conserva l’algoritmo upstream con la griglia corretta. |
| Rescale compresso multi-GPU | Aggiunti i kernel 1D per le lunghezze piccole e lo scratch condiviso per i plaintext senza buffer ausiliario; corretta la griglia del secondo passaggio 2D. I kernel matematici restano quelli upstream. |
| Packaging CMake | Esportata anche la dipendenza OpenFHE, ora necessaria per gli header pubblici upstream. Verificato il consumo della libreria installata dall’esempio offload. |
| Timeline NVTX e nostro trim del pool | Timeline upstream conservate; l'allocazione e il rilascio di slab tramite `GPUtrim` aggiornano i contatori NVTX. |
| Test bootstrap e interop locali | Adottate le suite upstream con confronto esatto; mantenuti i test specifici di offload/cache, memoria e regressioni del fork. Il vecchio documento di proposta Meta-BTS è marcato come storico. |
| Build CMake | Versione, opzioni, coverage, architetture e dipendenze upstream. Conservati prefix locali, compatibilità con subproject e installazione automatica della dipendenza mancante/incompatibile. |

## Estensioni del fork conservate

- Ciphertext `Offload`, `Reload`, `IsOffloaded` e ripristino automatico.
- `TrimGPUMemoryPool` e statistiche dei bucket del pool GPU.
- Budget VRAM LRU per chiavi di rotazione, plaintext e ciphertext; offload
  esplicito, pinning e contatori dei byte residenti.
- Budget per le precomputazioni bootstrap (`SetBootstrapCache`), snapshot
  pinned, caricamento per stage e riuso dei gruppi con politica MRU.
- Reload delle chiavi di rotazione direttamente da snapshot pinned.
- Limite opzionale della cache dei polinomi ausiliari tramite
  `FIDESLIB_AUX_POLY_CACHE_LIMIT`.
- Correzioni a sincronizzazione e riuso dei buffer, primo key-switch,
  concatenazione di operazioni, liberazione degli eventi CUDA e delle tabelle
  di puntatori, durata delle chiavi di parameter switching.
- Pool indicizzato per device fisico, contesti multi-GPU con device privi di
  special primes, NCCL come trasporto predefinito e controlli sui trasporti peer.
- Selezione dei device nei test tramite `--devices` e guard per NCCL/CUDA meno recenti.
- Esempio `examples/offload` e build con installazione interna al checkout.

I budget sono soft: il working set dell'operazione e gli oggetti pinned possono
superarli temporaneamente. Per chiavi di rotazione e bootstrap vanno impostati
prima di `LoadContext` per rendere disponibili gli snapshot. L'offload locale
dei ciphertext resta limitato alla base ordinaria; la base estesa viene gestita
dall'unload/serializzazione upstream e saltata dalla cache automatica.

## Verifica dell'integrazione

Build separata in `build/upstream-2.2.0`, senza sostituire la precedente build
né l'installazione OpenFHE 1.5.1. Le precedenti modifiche al submodule OpenFHE
sono conservate in `build/upstream-2.2.0/openfhe-1.5.1-preserved` e nel patch
`openfhe-1.5.1-local.patch`. Una copia durevole delle modifiche è conservata
anche nel repository Git del submodule al ref
`refs/backup/pre-upstream-2.2.0-openfhe`, indipendentemente dalla directory di build.
Il submodule principale segue ora il commit upstream.

I test di serializzazione upstream sono stati resi portabili: usano directory
temporanee isolate e rimosse automaticamente, invece del percorso locale
`/tmp/opencode` del progetto upstream.

La suite aggiunta `ForkCacheIntegrationTests.cu` combina budget nulli con
bootstrap standard, iterativo, StC-first, LT, complesso e sparse encapsulated;
verifica confronto CPU/GPU esatto, eviction ripetuta, export dopo offload e
composizione delle cache plaintext/ciphertext con il layout compresso.

Una regressione aggiuntiva confronta tre moltiplicazioni consecutive per plaintext
compressi a 512, 1024, 2048 e 4096 slot, coprendo entrambi i lati della soglia
NTT 1D/2D e i logaritmi pari/dispari.

Verifiche completate sulla build Release (`FIDESLIB_ARCH=90-real`), con
OpenFHE 1.6.1 ufficiale + patch completa, `WITH_NATIVEOPT=OFF`, GCC 13.3:

| Verifica | Esito |
| --- | --- |
| Build libreria, suite test e benchmark | Completata |
| Verifica estesa single-GPU, risultati consolidati per caso | **540 pass / 543 casi**, **3 skip** dei test P2P che richiedono più GPU |
| Riesecuzione mirata single-GPU | **132/132 pass**, inclusi tutti i **75 OpenFHECompatTests** e i **9 test di integrazione delle cache/compressione** |
| Due GPU, NCCL | **25/25 pass**: leak durante mult/bootstrap sparse iterativo, offload, primo key-switch, rotazioni/coniugazione, bootstrap iterativo e StC-first, tutte le regressioni della cache chiavi |
| Libreria installata → esempio offload | Configurazione, compilazione ed esecuzione completate |
| Inclusione come subproject CMake | Configurazione riuscita, compilatore del parent preservato |

I log e i report JUnit sono in `build/upstream-2.2.0/`:
`single-gpu-latest-results.xml`, `final-single-gpu.xml`,
`final-multi-gpu.xml`, `offload-example.log`.

La prima esecuzione estesa ha completato 534 casi, con 525 pass, 3 skip P2P
e 6 failure: quattro percorsi temporanei di serializzazione e due regressioni
NaN del rescale compresso. Tutti e sei sono stati corretti e superati nella
riesecuzione mirata da 132 casi. Il riepilogo consolidato usa l’ultima
esecuzione disponibile di ciascun caso e comprende anche i 9 nuovi test.
Non rappresenta una singola esecuzione monolitica dopo tutte le correzioni.
I 40 test `LLMTests` di timing non sono inclusi nella verifica funzionale.

## Verifica aggiuntiva durante l'aggiornamento di PyFIDESlib

Le regressioni Python hanno esposto altri due casi nel backend, corretti con test nativi:

- `dotProductPt_`: solo il plaintext usa l'indice compresso; i ciphertext conservano tutti
  gli `N` coefficienti. Usare l'indice compresso anche per i ciphertext produceva risultati
  errati con packing sparso e introduceva accessi concorrenti quando l'output era un input.
- `EvalAddManyInPlace` CPU: i cloni devono materializzare la propria copia prima della
  riduzione OpenFHE, altrimenti gli operandi condividono il payload e vengono sommati più volte.

I due nuovi test passano su una e due GPU (packing sparso e pieno, output alias di un input;
controllo bit-exact CPU/GPU per il prodotto). La suite nativa aggiuntiva ha verificato anche
75/75 test `OpenFHECompatTests` e i nove precedenti test d'integrazione delle cache.
I log/XML sono in `build/upstream-2.2.0/python-integration-*.{log,xml}`;
`python-integration-fixes-single.xml` contiene il primo giro con 84 pass e i due nuovi test
ancora da correggere nel setup, mentre `python-integration-final-{single,multi}.xml`
contiene il risultato definitivo 2/2 di entrambe le regressioni.
