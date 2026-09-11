# Changelog

## [0.12.0](https://github.com/ROCm/spur/compare/v0.11.0...v0.12.0) (2026-09-11)


### ⚠ BREAKING CHANGES

* **core:** two-field colon time values now mean minutes:seconds rather than hours:minutes, so they resolve to 1/60th of their previous duration. This affects `-t`/`--time` on sbatch, srun and salloc, and partition `MaxTime`/`DefaultTime`. Job scripts and site configs relying on the old reading should switch to the explicit `HH:MM:SS` form, which was and remains unambiguous. Bare `days-hours` now produces a real limit where it previously produced an unlimited job.

### Features

* node health checks as controller-submitted jobs ([#836](https://github.com/ROCm/spur/issues/836)) ([cee3f97](https://github.com/ROCm/spur/commit/cee3f974a2e865cef0ef867713a952f8627be556))
* **spur-cli:** add sinfo -R with node reason provenance (user/time) ([#860](https://github.com/ROCm/spur/issues/860)) ([018310d](https://github.com/ROCm/spur/commit/018310d8f97bac995f46597b343a0c1ed345d8cf))


### Bug Fixes

* **core:** read `-t MM:SS` as minutes:seconds and accept bare `days-hours` ([#811](https://github.com/ROCm/spur/issues/811)) ([127a99d](https://github.com/ROCm/spur/commit/127a99d349cf16c877600f55957811079811080b))
* **spur-cli:** honor sacct -j/--jobs job-ID filter ([#703](https://github.com/ROCm/spur/issues/703)) ([1fd143f](https://github.com/ROCm/spur/commit/1fd143fa8fcfd5eaa84d642a1a83cb785e07eab0))

## [0.11.0](https://github.com/ROCm/spur/compare/v0.10.0...v0.11.0) (2026-09-08)


### ⚠ BREAKING CHANGES

* **spurd:** stop injecting torch rendezvous env vars, match Slurm ([#793](https://github.com/ROCm/spur/issues/793))

### Features

* **cgroup:** add [cgroup] config and derive limits from the node allocation ([#762](https://github.com/ROCm/spur/issues/762)) ([bda405f](https://github.com/ROCm/spur/commit/bda405f9c39da5529f3a48fe5866aa52fb14cd88))
* **spur-cli:** add scontrol show assoc_mgr for limits against live usage ([#746](https://github.com/ROCm/spur/issues/746)) ([4c3719d](https://github.com/ROCm/spur/commit/4c3719d79f1968b7d34efd5df5c3407abaef73fe))
* **spur-cli:** honor --pty for srun steps in an allocation ([#820](https://github.com/ROCm/spur/issues/820)) ([de79428](https://github.com/ROCm/spur/commit/de794288e0e023aa4071502eb62ee437b99a37c0))
* **spurd:** file-back and live-stream srun step output ([#819](https://github.com/ROCm/spur/issues/819)) ([ee98dc3](https://github.com/ROCm/spur/commit/ee98dc3810f6532f12f1b4caabb9520dd6c87fb2))
* **spurd:** honor --container-image for srun --pty steps ([#845](https://github.com/ROCm/spur/issues/845)) ([2dfe3bb](https://github.com/ROCm/spur/commit/2dfe3bb0e1e7f14ed29cabf7bb9fb1956367eaa2))
* **spurd:** implement containerized srun job steps on bare-metal nodes ([#833](https://github.com/ROCm/spur/issues/833)) ([e71fc8b](https://github.com/ROCm/spur/commit/e71fc8b99c6d14e2b115121cb2eab2460893f5fe))


### Bug Fixes

* **spur-cli:** honor --export=ALL,VAR=val combined form and inline assignments ([#795](https://github.com/ROCm/spur/issues/795)) ([7093e3f](https://github.com/ROCm/spur/commit/7093e3fbdb54bb28b3f236bdb64f058897344c36))
* **spur-cli:** render squeue %L (TIME_LEFT) instead of ? ([#815](https://github.com/ROCm/spur/issues/815)) ([2369a96](https://github.com/ROCm/spur/commit/2369a96d872d749b7ff521ccaf5f9fd49e7322ef))
* **spurctld:** hold jobs whose QOS or association limits cannot be read ([#744](https://github.com/ROCm/spur/issues/744)) ([b58a7d8](https://github.com/ROCm/spur/commit/b58a7d83b7e9457ea5da0e103f6f0f433a5fa797))
* **spurctld:** make cni=calico work without the WireGuard mesh ([#812](https://github.com/ROCm/spur/issues/812)) ([c6d6894](https://github.com/ROCm/spur/commit/c6d6894495f3a01da1e5cbee8d7cc5e21ce6eba1))
* **spurd:** kernel-enforced GPU device isolation across every launch path ([#814](https://github.com/ROCm/spur/issues/814)) ([4bae73a](https://github.com/ROCm/spur/commit/4bae73aff0e55027e3274dd3a94045346f365323))
* **spurd:** kill container child before rootfs cleanup, honor --chdir in steps ([#838](https://github.com/ROCm/spur/issues/838)) ([f9655a6](https://github.com/ROCm/spur/commit/f9655a654b21c8482fb448f2fe4f6f9746a0f81c))
* **spurd:** start container jobs from the image's config.Env ([#676](https://github.com/ROCm/spur/issues/676)) ([3c873a5](https://github.com/ROCm/spur/commit/3c873a5120862b36d34a5bb16908da518094026e))
* **spurd:** stop injecting torch rendezvous env vars, match Slurm ([#793](https://github.com/ROCm/spur/issues/793)) ([fba350d](https://github.com/ROCm/spur/commit/fba350d75b03c5032067886f98e73a8c8c43f382))
* **tests:** pull e2e postgres image from mirror.gcr.io ([#796](https://github.com/ROCm/spur/issues/796)) ([3d950d7](https://github.com/ROCm/spur/commit/3d950d798a3646d18ad7a9fea8f7008df3ef080b))

## [0.10.0](https://github.com/ROCm/spur/compare/v0.9.1...v0.10.0) (2026-09-01)


### ⚠ BREAKING CHANGES

* **spurctld:** store job ids as 64-bit in the accounting database ([#753](https://github.com/ROCm/spur/issues/753))

### Features

* **spur-cli:** read cluster name from controller in scontrol show config ([#786](https://github.com/ROCm/spur/issues/786)) ([ddfb9ed](https://github.com/ROCm/spur/commit/ddfb9ed4b45b92f493bbc34f65dc4ba1d7755427))
* **spur-cli:** report requested placement and scheduling provenance in scontrol show job ([#761](https://github.com/ROCm/spur/issues/761)) ([9dbe37b](https://github.com/ROCm/spur/commit/9dbe37bb14b25f503fad4108747ae51adfc4a484))
* **spurctld:** add jobs_preempted event counter to scheduler stats ([#765](https://github.com/ROCm/spur/issues/765)) ([3bf032a](https://github.com/ROCm/spur/commit/3bf032a2cd4af0391f9cf7d31a626832ea4e29bf))


### Bug Fixes

* **auth:** align step RPC caller identity with JWT-bound job owner ([#760](https://github.com/ROCm/spur/issues/760)) ([19ff46a](https://github.com/ROCm/spur/commit/19ff46ab0d414ed63c1ce32ccd75ac86edbdc84b))
* fail rejected k8s jobs and enforce swap, burst-buffer, and scancel semantics ([#611](https://github.com/ROCm/spur/issues/611)) ([199488b](https://github.com/ROCm/spur/commit/199488be8a208bdf66efd259762b84fbd88218b1))
* **spur-cli:** accept Slurm's -n, -p and -P output flags on sacctmgr ([#669](https://github.com/ROCm/spur/issues/669)) ([dbe22bf](https://github.com/ROCm/spur/commit/dbe22bf9f01167daf81c2af9383453561231e410))
* **spur-cli:** show planned job/start and composite State= for plnd nodes ([#755](https://github.com/ROCm/spur/issues/755)) ([c92d0dd](https://github.com/ROCm/spur/commit/c92d0dd13a4541c8390133baa28166a13ecea256))
* **spur-cli:** stop sprio and sshare panicking on every invocation ([#697](https://github.com/ROCm/spur/issues/697)) ([233029a](https://github.com/ROCm/spur/commit/233029a66acd6654a1c5a878e99f161ed6f1ee55))
* **spurctld:** address review comments on jobs_preempted counter ([#766](https://github.com/ROCm/spur/issues/766)) ([4dc6710](https://github.com/ROCm/spur/commit/4dc67106cd1bbd899f95fba116fbe5d1852798d9))
* **spurctld:** apply MaxWall as the default time limit for a job with no -t ([#728](https://github.com/ROCm/spur/issues/728)) ([4039d68](https://github.com/ROCm/spur/commit/4039d68c197d46e7a4edfe4d07e4d6cfdd57d729))
* **spurctld:** categorize dispatch confirmation failure reasons ([#717](https://github.com/ROCm/spur/issues/717)) ([87787ae](https://github.com/ROCm/spur/commit/87787ae8d55dfbdde46bb685f696e07b4f56955d))
* **spurctld:** keep one unresponsive node from stalling all scheduling ([#776](https://github.com/ROCm/spur/issues/776)) ([4f933b3](https://github.com/ROCm/spur/commit/4f933b3f278bf42b3e5bd3c75b55a2f4eb92a8e2))
* **spurctld:** report array task throttle reason ([#720](https://github.com/ROCm/spur/issues/720)) ([b5c6ece](https://github.com/ROCm/spur/commit/b5c6ece6579d25be8bf28e26a4875494bd0beacd))
* **spurctld:** store job ids as 64-bit in the accounting database ([#753](https://github.com/ROCm/spur/issues/753)) ([180c896](https://github.com/ROCm/spur/commit/180c896d8a1fdfe74ffe3a63c18d978a4a89dc2d))
* **spurctld:** type the EXPLAIN fixture's job id as JobId ([#769](https://github.com/ROCm/spur/issues/769)) ([5c1f662](https://github.com/ROCm/spur/commit/5c1f6620e60db3262c3d9fcbedfd08987f73f417))


### Performance

* **spurctld:** index the GrpWall consumption window ([#752](https://github.com/ROCm/spur/issues/752)) ([cf9d0a9](https://github.com/ROCm/spur/commit/cf9d0a9a83c59e5dd5be58847f40bac251662076))

## [0.9.1](https://github.com/ROCm/spur/compare/v0.9.0...v0.9.1) (2026-08-25)


### Bug Fixes

* **spur-cli:** accept Slurm's --noconvert on squeue and sinfo ([#647](https://github.com/ROCm/spur/issues/647)) ([bfc288c](https://github.com/ROCm/spur/commit/bfc288c11aaec5fa007e5b373543aa193dd562ff))
* **spur-sched:** protect large multi-node jobs from backfill starvation ([#732](https://github.com/ROCm/spur/issues/732)) ([1b2826b](https://github.com/ROCm/spur/commit/1b2826bf1ca0313efa7d08f97b152a55fe1c7cb9))
* **spurd:** stop warning when no spur.conf exists at the default path ([#631](https://github.com/ROCm/spur/issues/631)) ([54e1eaf](https://github.com/ROCm/spur/commit/54e1eaf17692c667ba3dfae8595810b4a9922d24))

## [0.9.0](https://github.com/ROCm/spur/compare/v0.8.0...v0.9.0) (2026-08-24)


### ⚠ BREAKING CHANGES

* **spurctld:** enforce reservation management privileges server-side ([#726](https://github.com/ROCm/spur/issues/726))
* **spurctld:** enforce QOS and association submit-time limits ([#670](https://github.com/ROCm/spur/issues/670))

### Features

* **auth:** authenticate callers of the agent (spurd) surface ([#641](https://github.com/ROCm/spur/issues/641)) ([54f3763](https://github.com/ROCm/spur/commit/54f3763b9bc9ca8bdf3d094364f225bbf282bd2a))
* **auth:** authenticate control-plane callers (identity spine) ([#640](https://github.com/ROCm/spur/issues/640)) ([4a5201c](https://github.com/ROCm/spur/commit/4a5201ca9218ad5ba5f06860a917ca313c27c990))
* **spur-cli:** filter squeue by node with -w/--nodelist ([#628](https://github.com/ROCm/spur/issues/628)) ([42a1519](https://github.com/ROCm/spur/commit/42a15193052305a53bec9612d0942638d3ed503d))
* **spur-core:** WireGuard mesh + k0s lifecycle CLI for node add/remove ([#719](https://github.com/ROCm/spur/issues/719)) ([bd3121d](https://github.com/ROCm/spur/commit/bd3121d0504a1229a3d4d9cf885a2081b1bc8da7))
* **spurctld:** add job submission validation hook (shell + lua) ([#565](https://github.com/ROCm/spur/issues/565)) ([18e5653](https://github.com/ROCm/spur/commit/18e565325bab0d91670eb89930b3f5f5e46d3e8a))
* **spurctld:** enforce QOS and association submit-time limits ([#670](https://github.com/ROCm/spur/issues/670)) ([21017b6](https://github.com/ROCm/spur/commit/21017b6d0b12cc98477474b54dcd9f6f4ae53c28))
* **spurctld:** implement scontrol requeue and requeuehold ([#629](https://github.com/ROCm/spur/issues/629)) ([b3c9b0f](https://github.com/ROCm/spur/commit/b3c9b0f5245ebe47eb2931300c5bbfa2f65fedb4))
* **spurctld:** k0s partial-Ready and online worker add/remove ([#655](https://github.com/ROCm/spur/issues/655)) ([174fa20](https://github.com/ROCm/spur/commit/174fa206707fde228b72f4c2b90b60009d9f58d5))
* **spurctld:** QOS preemption hierarchy and minimum exempt time ([#683](https://github.com/ROCm/spur/issues/683)) ([30333b6](https://github.com/ROCm/spur/commit/30333b6e94458164dd1574b73de69fa3d59e62fe))
* **spurctld:** reap abandoned interactive allocations via client keepalive ([#594](https://github.com/ROCm/spur/issues/594)) ([6a0568b](https://github.com/ROCm/spur/commit/6a0568b2c88c9a3fda3b44a3995692cf41716eec))
* **spurctld:** record preemption provenance on job state ([#700](https://github.com/ROCm/spur/issues/700)) ([513d319](https://github.com/ROCm/spur/commit/513d319cbaf5af7b2f8a46272c152457fb200821))
* **spurctld:** record reservation admin actions in accounting txn log ([#698](https://github.com/ROCm/spur/issues/698)) ([d0909e5](https://github.com/ROCm/spur/commit/d0909e57dcb45eae27c8815d3800d75700d5cea1))


### Bug Fixes

* **deps:** bump h2 to 0.4.16 to resolve RUSTSEC-2026-0258 ([#671](https://github.com/ROCm/spur/issues/671)) ([c6e8e7b](https://github.com/ROCm/spur/commit/c6e8e7b21a3d1e9a78053236a6fd8c23da0e16a9))
* **security:** enforce caller identity on job-mutation and agent RPCs ([#686](https://github.com/ROCm/spur/issues/686)) ([3c0e7af](https://github.com/ROCm/spur/commit/3c0e7af1d909e33b2b3c301e3df43538c356fd85))
* **security:** fail closed on account ACLs and the required-mode signing key ([#688](https://github.com/ROCm/spur/issues/688)) ([7bddd79](https://github.com/ROCm/spur/commit/7bddd79f8f1d8806eb04a6ad18c71c2e8778b9da))
* **security:** gate control-plane mutation RPCs on verified admin ([#685](https://github.com/ROCm/spur/issues/685)) ([d500bad](https://github.com/ROCm/spur/commit/d500badd37d75d8b5552d2dd1bb25db664325352))
* **security:** harden the k8s operator plane tenancy boundary ([#687](https://github.com/ROCm/spur/issues/687)) ([87e1135](https://github.com/ROCm/spur/commit/87e1135d94511fa8edb4de5d91b7612ba99decec))
* **security:** refuse root execution, default the REST API off, gate the admin kubeconfig ([#637](https://github.com/ROCm/spur/issues/637)) ([ab802e4](https://github.com/ROCm/spur/commit/ab802e4260ab73866c77f24457f48219661d771e))
* **spur-cli:** accept slurm-format reservation durations ([#626](https://github.com/ROCm/spur/issues/626)) ([9697ed6](https://github.com/ROCm/spur/commit/9697ed6a191250ffd49fd30bbe4949e9c38cad6b))
* **spur-cli:** accept Slurm's full time grammar for --begin and --deadline ([#556](https://github.com/ROCm/spur/issues/556)) ([10971ec](https://github.com/ROCm/spur/commit/10971ec1dc640356f8bf0870cedd46d56b346968))
* **spur-cli:** resolve first allocated node from compressed hostlist ([#589](https://github.com/ROCm/spur/issues/589)) ([eee669e](https://github.com/ROCm/spur/commit/eee669e69b81047e34124efffab9bcfea6fda526))
* **spur-cli:** support ALL in node subcommands ([#568](https://github.com/ROCm/spur/issues/568)) ([4ad42c9](https://github.com/ROCm/spur/commit/4ad42c9df12e69d069a67b33ad62f5140c2cff56))
* **spur-core:** bound array size before expansion and use checked step increment ([#646](https://github.com/ROCm/spur/issues/646)) ([24cf38c](https://github.com/ROCm/spur/commit/24cf38c6c9f3183c60c41ade54545d0cf2e727bb)), closes [#645](https://github.com/ROCm/spur/issues/645)
* **spur-core:** cap hostlist expansion size to prevent controller OOM ([#679](https://github.com/ROCm/spur/issues/679)) ([8490669](https://github.com/ROCm/spur/commit/849066970e6f608c862b8e41a5a46f31ab8a9827))
* **spur-core:** deny GPU visibility when zero devices are allocated ([#648](https://github.com/ROCm/spur/issues/648)) ([323ea7e](https://github.com/ROCm/spur/commit/323ea7ee2f66f15d95ed8a76035bab7250ae3b66))
* **spur-core:** emit node-local load balancing for HA mesh k0s clusters ([#708](https://github.com/ROCm/spur/issues/708)) ([36213a1](https://github.com/ROCm/spur/commit/36213a19c8932e1fc9b8c3746ac1c4eb09eabc58))
* **spur-core:** ignore quoted srun tokens in batch step detection ([#627](https://github.com/ROCm/spur/issues/627)) ([1765105](https://github.com/ROCm/spur/commit/17651055ea7e988f7429587beb7ac915806768c7))
* **spur-core:** prevent stray ']' from merging hosts in split_top_level ([#643](https://github.com/ROCm/spur/issues/643)) ([5ac18f0](https://github.com/ROCm/spur/commit/5ac18f01dbe2705e2e0092c032dae0176523caa6))
* **spur-core:** reject hostlist patterns with ']' before '[' ([#639](https://github.com/ROCm/spur/issues/639)) ([bdd2528](https://github.com/ROCm/spur/commit/bdd252880ce7b061cbf535e89287f6787f0460c8))
* **spur-core:** use checked arithmetic in time parsers to prevent overflow ([#680](https://github.com/ROCm/spur/issues/680)) ([4cc3da8](https://github.com/ROCm/spur/commit/4cc3da843d7782774b616b2cdf7fc1e0d2cde777))
* **spurctld:** adopt each node's real WireGuard address as its k0s mesh IP ([#699](https://github.com/ROCm/spur/issues/699)) ([a2fe221](https://github.com/ROCm/spur/commit/a2fe22100783af54d3f8420961353888a85b2789))
* **spurctld:** cool down a node that rejects a dispatch as resources-unavailable ([#584](https://github.com/ROCm/spur/issues/584)) ([e19a7a1](https://github.com/ROCm/spur/commit/e19a7a17f31f8829bd7f49da94a25cbb426dd6f7))
* **spurctld:** count distinct nodes for the QoS group node limit ([#709](https://github.com/ROCm/spur/issues/709)) ([#714](https://github.com/ROCm/spur/issues/714)) ([a3da921](https://github.com/ROCm/spur/commit/a3da92103b3c56cb82002d7dedef4f7ee11bcda8))
* **spurctld:** enforce partition walltime limits and validate time config ([#593](https://github.com/ROCm/spur/issues/593)) ([172bf8f](https://github.com/ROCm/spur/commit/172bf8f90aea309a90b8bf17bad2ff530ea23cb9))
* **spurctld:** enforce QOS GrpWall at scheduling admission ([#675](https://github.com/ROCm/spur/issues/675)) ([5a648bd](https://github.com/ROCm/spur/commit/5a648bd24efea7d126e510ed4f2f50959d3f6d0d))
* **spurctld:** enforce reservation management privileges server-side ([#726](https://github.com/ROCm/spur/issues/726)) ([5fcf4fa](https://github.com/ROCm/spur/commit/5fcf4faa193d5e7ce22b1838f7f560a7882893d1))
* **spurctld:** hold off marking nodes down right after leadership change ([#624](https://github.com/ROCm/spur/issues/624)) ([33b65f6](https://github.com/ROCm/spur/commit/33b65f60f4cfcc7895b8dd9836e3d9123676360f))
* **spurctld:** preserve node allocations during recovery ([#613](https://github.com/ROCm/spur/issues/613)) ([42cfe92](https://github.com/ROCm/spur/commit/42cfe92bd8383f343fa39d9668dab37720d0b43a))
* **spurctld:** QOS priority not applied to job priority; preemption never triggers ([#652](https://github.com/ROCm/spur/issues/652)) ([48d28bb](https://github.com/ROCm/spur/commit/48d28bb18c8d4cacdec6a041ee2145afebe1c8aa))
* **spurctld:** reclaim agent allocations the controller no longer tracks on that node ([#623](https://github.com/ROCm/spur/issues/623)) ([0d8b8cb](https://github.com/ROCm/spur/commit/0d8b8cb0756e024a2615adf5381a014f67bc80e4))
* **spurctld:** reject submits from users with no account association ([#706](https://github.com/ROCm/spur/issues/706)) ([d6001b4](https://github.com/ROCm/spur/commit/d6001b473747ea18883b0b4336959c951ace6865))
* **spurctld:** report BeginTime while --begin defers a job ([#555](https://github.com/ROCm/spur/issues/555)) ([c20a051](https://github.com/ROCm/spur/commit/c20a05127b2f225876cf8445ac5a53dbb4be03b0))
* **spurctld:** tag in-pass-blocked jobs so pending reason is never None ([#620](https://github.com/ROCm/spur/issues/620)) ([831390d](https://github.com/ROCm/spur/commit/831390dc6922bde598600c082755bfa8caf00b15))
* **spurd:** close container sync-pipe fds exactly once ([#656](https://github.com/ROCm/spur/issues/656)) ([e1706c6](https://github.com/ROCm/spur/commit/e1706c6aead0e142d45279d4629cfc5968163171))
* **spurd:** purge k0s join token on teardown even when reset fails ([#674](https://github.com/ROCm/spur/issues/674)) ([620070a](https://github.com/ROCm/spur/commit/620070ac2cbc37917ff7e0f94b742ec31a68f31c))
* **spurd:** restore supplementary groups on nsenter exec/overlap paths ([#633](https://github.com/ROCm/spur/issues/633)) ([cd24487](https://github.com/ROCm/spur/commit/cd24487639589d8f027628364ae80ec8f57309a1))
* **tenancy:** clamp non-admin priority and redact cross-tenant job info ([#663](https://github.com/ROCm/spur/issues/663), [#664](https://github.com/ROCm/spur/issues/664)) ([#666](https://github.com/ROCm/spur/issues/666)) ([1e535f2](https://github.com/ROCm/spur/commit/1e535f2cea5891dea17c20fe8171cd022ee0ac08))
* use checked arithmetic for CIDR prefixes and token TTLs ([#716](https://github.com/ROCm/spur/issues/716)) ([fa629e8](https://github.com/ROCm/spur/commit/fa629e86568f9b95eb2a23f4aaecbe5b5b1eda1e))

## [0.8.0](https://github.com/ROCm/spur/compare/v0.7.0...v0.8.0) (2026-08-11)


### ⚠ BREAKING CHANGES

* **spur-cli:** non-privileged users can no longer manage reservations via the CLI; scripts relying on the old permissive behavior will fail.

### Features

* **mpi:** multi-node PMIx prepare/join with TCP modex exchange ([#561](https://github.com/ROCm/spur/issues/561)) ([2eac05d](https://github.com/ROCm/spur/commit/2eac05d103999e759acef1e93514aa18590cb49f))
* **spur-cli:** restrict reservation mgmt to root and sudo/wheel users ([#595](https://github.com/ROCm/spur/issues/595)) ([8849ce1](https://github.com/ROCm/spur/commit/8849ce1cf28fe6111376a8924b6e001f43bc6313))
* **spur-cli:** runtime partition CRUD via Raft WAL with scontrol reconfigure ([#425](https://github.com/ROCm/spur/issues/425)) ([05cf9a6](https://github.com/ROCm/spur/commit/05cf9a6469b29e02c0b791d101b21d9a1c00cbc5))
* **spurctld:** degrade k0s cluster on provisioning timeout ([#585](https://github.com/ROCm/spur/issues/585)) ([0487721](https://github.com/ROCm/spur/commit/04877214e1650827b1b8a9cce5cdb63a5551be44))
* **spurctld:** export k0s cluster and node metrics at /metrics/k8s ([#606](https://github.com/ROCm/spur/issues/606)) ([857ad7e](https://github.com/ROCm/spur/commit/857ad7ec70d5cda9bca30fef85e539f6f226355e))
* **spurctld:** scope k0s cluster to a subset of nodes ([#577](https://github.com/ROCm/spur/issues/577)) ([4b45256](https://github.com/ROCm/spur/commit/4b452568b31f976a1249ceca7a794dd4c1a4311c))


### Bug Fixes

* **spur-cli:** compress sinfo NODELIST as a Slurm-style hostlist ([#574](https://github.com/ROCm/spur/issues/574)) ([bce95e1](https://github.com/ROCm/spur/commit/bce95e1c36d3790c60c0412e762fb7634f8d8948))
* **spur-cli:** reject unresolved usernames ([#569](https://github.com/ROCm/spur/issues/569)) ([0b10e08](https://github.com/ROCm/spur/commit/0b10e08c01fe689684c013bf0f8ec77a2466f55d))
* **spur-net:** cache OCI layers under the resolved image directory ([#507](https://github.com/ROCm/spur/issues/507)) ([4f3fa31](https://github.com/ROCm/spur/commit/4f3fa31c412a3cf7812b189d059b54c41a82e7c8))
* **spur-net:** support gzip, zstd, and uncompressed OCI layers ([#578](https://github.com/ROCm/spur/issues/578)) ([1f5f4c7](https://github.com/ROCm/spur/commit/1f5f4c75f30c57f8ee668a657c92318797e5af10))
* **spur-sched:** allocate CPUs and cap nodes when tasks fewer than nodes ([#570](https://github.com/ROCm/spur/issues/570)) ([877b274](https://github.com/ROCm/spur/commit/877b2748e9876cfb4cca73d246c560ea501bf136))
* **spurctld:** evict aged terminal jobs to bound controller memory ([#583](https://github.com/ROCm/spur/issues/583)) ([a261284](https://github.com/ROCm/spur/commit/a261284eb66541b9b99b558e7019b594965519d8))
* **spurctld:** filter reservations by name in scontrol show ([#581](https://github.com/ROCm/spur/issues/581)) ([2bd8a4b](https://github.com/ROCm/spur/commit/2bd8a4b2826e95a038bdd5dbc080a4286ec912c3))
* **spurctld:** report wall-time expiry as TIMEOUT ([#554](https://github.com/ROCm/spur/issues/554)) ([4b9a138](https://github.com/ROCm/spur/commit/4b9a138d6deb9ae3b745d5b31b90d6275fafa915))
* **spurctld:** return unavailable when accounting startup fails ([#598](https://github.com/ROCm/spur/issues/598)) ([b81c26b](https://github.com/ROCm/spur/commit/b81c26b727e97c52853ccd161c920103cf15c9bf))
* **spurd:** reclaim stranded interactive allocations via heartbeat ([#580](https://github.com/ROCm/spur/issues/580)) ([2dea550](https://github.com/ROCm/spur/commit/2dea5505fe23e2e286a7dc3966239ecaef979199))

## [0.7.0](https://github.com/ROCm/spur/compare/v0.6.0...v0.7.0) (2026-08-04)


### Refactors

* **spurd:** drop --comm-address alias and SPUR_COMM_ADDRESS ([#562](https://github.com/ROCm/spur/issues/562)) ([9f4506d](https://github.com/ROCm/spur/commit/9f4506d442408719abb8924258022615f9a5e1b1))


### Features

* **mpi:** add single-node PMIx MPI via dlopen plugin ([#505](https://github.com/ROCm/spur/issues/505)) ([3fbc62a](https://github.com/ROCm/spur/commit/3fbc62a95f56f4a71b6571091970c91f7863c2b6))
* **mpi:** fix single-node PMIx modex and mpirun launch ([#528](https://github.com/ROCm/spur/issues/528)) ([83c8b4c](https://github.com/ROCm/spur/commit/83c8b4c1911b460c39090a8617ffb800b878963f))
* **quota:** Kubernetes quota-enforcement — M1 (tenancy + hard quotas + scoped kubeconfig) ([#475](https://github.com/ROCm/spur/issues/475)) ([aeb21cd](https://github.com/ROCm/spur/commit/aeb21cd4211b692b2970862d9953580fddc4fb8f))
* **spur-cli:** accept --ntasks-per-node in srun ([#534](https://github.com/ROCm/spur/issues/534)) ([b097540](https://github.com/ROCm/spur/commit/b0975402e960523304f5db202678cbd9efba9988))
* **spur-net:** normalize node comm addresses for agent reachability ([#557](https://github.com/ROCm/spur/issues/557)) ([8c53f3e](https://github.com/ROCm/spur/commit/8c53f3e1d18be65bb683dca8a35d3463c1b4c622))
* **spurctld:** authorize k0s cluster ops and scope kubeconfigs per user ([#560](https://github.com/ROCm/spur/issues/560)) ([f9a28be](https://github.com/ROCm/spur/commit/f9a28be7d0f75b93cbc3f9203399027af4856d96))
* **spurctld:** multi-control-plane HA for native k0s clusters ([#529](https://github.com/ROCm/spur/issues/529)) ([0e5262d](https://github.com/ROCm/spur/commit/0e5262d19ab0c427244d5b1f5131a88634652669))


### Bug Fixes

* **accounting:** preserve unset fields on sacctmgr modify ([#527](https://github.com/ROCm/spur/issues/527)) ([6275ea8](https://github.com/ROCm/spur/commit/6275ea893af3426c7d047304f0b3b14d19fcd1f7))
* **auth:** enforce job ownership on exec, attach, and output streaming ([#551](https://github.com/ROCm/spur/issues/551)) ([367838e](https://github.com/ROCm/spur/commit/367838e5b52c6c926be75dcd4087720cd0b3ed2f))
* **build:** sync Cargo.lock spur-mpi-pmix version to 0.6.0 ([#523](https://github.com/ROCm/spur/issues/523)) ([8d9522d](https://github.com/ROCm/spur/commit/8d9522dfbb110b231896a09fbf6d5300a635e1e3))
* **dispatch:** back off between launch retries and drain unusable nodes ([#538](https://github.com/ROCm/spur/issues/538)) ([c267377](https://github.com/ROCm/spur/commit/c267377f0c7f6bb670d1296639aea93aef7bdac5))
* **spur-cli:** accept --qos and SALLOC_QOS in salloc ([#540](https://github.com/ROCm/spur/issues/540)) ([0221042](https://github.com/ROCm/spur/commit/02210426638ea81c426091c4eaec0dddbbf59b4a))
* **spur-cli:** accept multi-node hostlists in scontrol update NodeName ([#549](https://github.com/ROCm/spur/issues/549)) ([3120d5c](https://github.com/ROCm/spur/commit/3120d5ce8888d8ad68e1f83ddc5cf3006d96ac59))
* **spur-cli:** apply sinfo --states filter to node query ([#533](https://github.com/ROCm/spur/issues/533)) ([a92bf5e](https://github.com/ROCm/spur/commit/a92bf5e0d3360914b41de1f1917a67190f7344bb))
* **spur-cli:** default sbatch --ntasks to one task per node ([#522](https://github.com/ROCm/spur/issues/522)) ([316888d](https://github.com/ROCm/spur/commit/316888d382e9c23ba94f996fb1d94d8c385c0f19))
* **spur-cli:** default srun and salloc --ntasks to one task per node ([#532](https://github.com/ROCm/spur/issues/532)) ([f478b20](https://github.com/ROCm/spur/commit/f478b2019aa4baf1423048d3e61eb91ddb058303))
* **spur-cli:** honor format= in sacctmgr show account ([#519](https://github.com/ROCm/spur/issues/519)) ([ff85e9b](https://github.com/ROCm/spur/commit/ff85e9b2946d7c10a7745b8425e19a7a21fd4aff))
* **spur-cli:** honor SPUR_* env-var twins in sbatch ([#526](https://github.com/ROCm/spur/issues/526)) ([bdc4367](https://github.com/ROCm/spur/commit/bdc4367fcc7282110e97d6276de220fdf159654a))
* **spur-cli:** scale default sbatch --ntasks by --ntasks-per-node ([#531](https://github.com/ROCm/spur/issues/531)) ([b0ea620](https://github.com/ROCm/spur/commit/b0ea620967765dab2d50ab105dad125d4b00f3ff))
* **spur-cli:** sort squeue output and honor -S/--sort ([#524](https://github.com/ROCm/spur/issues/524)) ([7a0b2c9](https://github.com/ROCm/spur/commit/7a0b2c9b70b506e0f54b9a9781b88237f524da13))
* **spur-cli:** target the -w node when attaching an interactive PTY ([#516](https://github.com/ROCm/spur/issues/516)) ([dd5f6e3](https://github.com/ROCm/spur/commit/dd5f6e37c27bb199e074f656bc73360e87d1620d))
* **spur-core:** default JobSpec.pty so pre-0.6 Raft logs replay ([#509](https://github.com/ROCm/spur/issues/509)) ([d754118](https://github.com/ROCm/spur/commit/d754118b23c68b8fff876f29dff442f3b12a65be))
* **spurctld:** confirm batch dispatch on every node before Running ([#550](https://github.com/ROCm/spur/issues/550)) ([222d0f4](https://github.com/ROCm/spur/commit/222d0f4cfb302a3fac9e474fb328f0c1bf50a7dd))
* **spurctld:** delete sacctmgr users atomically ([#451](https://github.com/ROCm/spur/issues/451)) ([5c3ddce](https://github.com/ROCm/spur/commit/5c3ddce07da2d488e6d5c7b6614d78ba6990d94f))
* **spurctld:** distinguish missing vs unavailable QOS in validation errors ([#508](https://github.com/ROCm/spur/issues/508)) ([6a2b9bc](https://github.com/ROCm/spur/commit/6a2b9bc4c2ca6aaaf958a475e12ac6093027e30a))
* **spurctld:** exclude k0s-reserved nodes from batch scheduling ([#558](https://github.com/ROCm/spur/issues/558)) ([d4767da](https://github.com/ROCm/spur/commit/d4767da60d325070a8a4aa6881c1b664c8e97646))
* **spurctld:** serve read RPCs from local state when no Raft leader ([#506](https://github.com/ROCm/spur/issues/506)) ([93621c7](https://github.com/ROCm/spur/commit/93621c74cc9a356a15370d1db90d7562bd43dab4))
* **spurctld:** unify pending job classification ([#500](https://github.com/ROCm/spur/issues/500)) ([96c37de](https://github.com/ROCm/spur/commit/96c37de331441dcf7e1d84e164756c89ad2b47dc))
* **spurd:** run sbatch batch script once per node, not once per task ([#552](https://github.com/ROCm/spur/issues/552)) ([a1be3b7](https://github.com/ROCm/spur/commit/a1be3b7f28deb0d93560b1be0cde769260b05eb4))

## [0.6.0](https://github.com/ROCm/spur/compare/v0.5.1...v0.6.0) (2026-07-27)


### Features

* **cluster:** SPUR-native k0s clusters via spur k8s up/down + mesh-native CNI ([#432](https://github.com/ROCm/spur/issues/432)) ([7c95efe](https://github.com/ROCm/spur/commit/7c95efe2ca34cede227a0fa2e6592173118f5c8e))
* **spur-cli:** accept --qos for srun ([#481](https://github.com/ROCm/spur/issues/481)) ([765f319](https://github.com/ROCm/spur/commit/765f31956f413afe12fc674934557a91ddf4bbb4))
* **spur-cli:** interactive PTY sessions via srun --pty and sattach ([#482](https://github.com/ROCm/spur/issues/482)) ([a1b28fb](https://github.com/ROCm/spur/commit/a1b28fb046dc861060b512e9e4cce6450a681786))
* **spur-cli:** persist and display account-level grp_tres allocation ([#477](https://github.com/ROCm/spur/issues/477)) ([b1fa1ee](https://github.com/ROCm/spur/commit/b1fa1ee9f02d5afe5b5789d67de9cea0df5a74be))
* **spur-cli:** support exclusive srun allocations ([#496](https://github.com/ROCm/spur/issues/496)) ([c56b873](https://github.com/ROCm/spur/commit/c56b87301f305c50990b07e9098c3d83961190fb))
* **spur-cli:** support nodefile-based nodelists ([#467](https://github.com/ROCm/spur/issues/467)) ([9947815](https://github.com/ROCm/spur/commit/9947815ef70522463824a0b53bf4c5ae078c3ce4))
* **spur-core:** explicit GPU request modeling with greedy distribution ([#502](https://github.com/ROCm/spur/issues/502)) ([996cb54](https://github.com/ROCm/spur/commit/996cb54e90ce22edec051a603326e241a9dd840e))
* **spur-sched:** make nodelist selection additive ([#472](https://github.com/ROCm/spur/issues/472)) ([a4a29e6](https://github.com/ROCm/spur/commit/a4a29e62f699ea692c93944f4955f53502de7070))
* **spurctld:** auto-derive raft node_id from controller.peers position ([#459](https://github.com/ROCm/spur/issues/459)) ([6967f29](https://github.com/ROCm/spur/commit/6967f29009e0635558018106e095fdc45c3042c9))
* **spurctld:** report node-resolved job output path in scontrol ([#488](https://github.com/ROCm/spur/issues/488)) ([bb7e9f8](https://github.com/ROCm/spur/commit/bb7e9f85d6e0b231f83eef70ba24e2ebdfcd0f7d))
* **spurctld:** show valid accounts in association errors ([#446](https://github.com/ROCm/spur/issues/446)) ([db7bef1](https://github.com/ROCm/spur/commit/db7bef1dca12beb7cda2e86ae70ef45f843dadde))


### Bug Fixes

* **accounting:** filter users by name ([#447](https://github.com/ROCm/spur/issues/447)) ([b3e16a1](https://github.com/ROCm/spur/commit/b3e16a1418d64a10b8e769c9a73d3725ff2090db))
* **oci:** key container images by canonical reference ([#497](https://github.com/ROCm/spur/issues/497)) ([5bc467c](https://github.com/ROCm/spur/commit/5bc467c27c5f4784fdf27f5266d2a03bb1b93700))
* **spur-cli:** display node features in sinfo ([#493](https://github.com/ROCm/spur/issues/493)) ([f3ee133](https://github.com/ROCm/spur/commit/f3ee133b8ba12406ade6578cf10efed6c67e80e8))
* **spur-cli:** limit user cancellation to active jobs ([#483](https://github.com/ROCm/spur/issues/483)) ([42750ac](https://github.com/ROCm/spur/commit/42750ace253366355255957fea212a818efc95ac))
* **spur-cli:** skip terminal jobs in filter-based scancel ([#462](https://github.com/ROCm/spur/issues/462)) ([c615aa5](https://github.com/ROCm/spur/commit/c615aa5e870e3a9d27250a8b2136d330ea1d0597))
* **spur-core:** run hooks from /tmp when work_dir can't be entered to avoid draining nodes ([#503](https://github.com/ROCm/spur/issues/503)) ([ec97012](https://github.com/ROCm/spur/commit/ec97012f2238aa9fe34479b2afcca7746666455a))
* **spur-sched:** spread equal-weight jobs across idle nodes ([#453](https://github.com/ROCm/spur/issues/453)) ([c41f188](https://github.com/ROCm/spur/commit/c41f188450011df8d8a6510f63442913d0cf0176))
* **spurctld:** drop stale completion reports from preempted-and-requeued runs ([#474](https://github.com/ROCm/spur/issues/474)) ([e4e3288](https://github.com/ROCm/spur/commit/e4e32889145b6e0221fd0350d218e9e5e96311b9))
* **spurctld:** enforce ownership on reservation delete and update ([#463](https://github.com/ROCm/spur/issues/463)) ([d7fc6cd](https://github.com/ROCm/spur/commit/d7fc6cd76172501f4afd2a0e82f7006dba5c868b))
* **spurctld:** prevent single scheduling pass from over-subscribing QOS/account aggregate limits ([#478](https://github.com/ROCm/spur/issues/478)) ([f656944](https://github.com/ROCm/spur/commit/f65694402550cda8d91e4939d5f559612d2b8af1))
* **spurctld:** reject job QOS not authorized for the submitting association ([#490](https://github.com/ROCm/spur/issues/490)) ([59b7180](https://github.com/ROCm/spur/commit/59b718030d70b0d4541b2f89a591d730dddec8e1))
* **spurctld:** schedule jobs across partition OR-lists ([#468](https://github.com/ROCm/spur/issues/468)) ([90986b5](https://github.com/ROCm/spur/commit/90986b5d97226155baa5f41e9d5ab7fa7a73b964))
* **spurctld:** update upsert_account test call for grp_tres arity ([#485](https://github.com/ROCm/spur/issues/485)) ([1ed491a](https://github.com/ROCm/spur/commit/1ed491a6ca56005a57bbec5706f4c0b01be51049))
* **spurd:** release GPU reservations stranded mid-launch ([#489](https://github.com/ROCm/spur/issues/489)) ([c5b52cd](https://github.com/ROCm/spur/commit/c5b52cd948ba4ce066d353b4d8a63d32a6113895))
* **spur:** reclaim node resources when completion and eviction desync ([#498](https://github.com/ROCm/spur/issues/498)) ([9c2098c](https://github.com/ROCm/spur/commit/9c2098ce013481b4d2799e52463c7751f4b2a864))
* **srun:** fan out tasks across nodes in standalone and step mode ([#473](https://github.com/ROCm/spur/issues/473)) ([c74abac](https://github.com/ROCm/spur/commit/c74abac900ff64ae18dd349db212869f6b5cecd9))

## [0.5.1](https://github.com/ROCm/spur/compare/v0.5.0...v0.5.1) (2026-07-17)


### Bug Fixes

* **spur-cli:** hide unregistered nodes from sinfo ([#465](https://github.com/ROCm/spur/issues/465)) ([3052801](https://github.com/ROCm/spur/commit/30528013a78c70c017d61f0fd4ec49b22bbfec59))
* **spur-cli:** wire format engine into sacctmgr show qos ([#457](https://github.com/ROCm/spur/issues/457)) ([a422f51](https://github.com/ROCm/spur/commit/a422f516d286ab665e949d45104deff698aa32d5))
* **spur-core:** derive version from cargo version plus git sha ([#458](https://github.com/ROCm/spur/issues/458)) ([055336f](https://github.com/ROCm/spur/commit/055336fd3160f6bc526b858c7c2f29ab6cde8220))
* **spur-core:** enforce node and memory limits in QOS per-user checks ([#427](https://github.com/ROCm/spur/issues/427)) ([7595e9e](https://github.com/ROCm/spur/commit/7595e9e27c0e7a90e0eddda34f6a22cadf3a19fd))
* **spurctld:** close QOS enforcement bypass for jobs submitted without a QOS ([#452](https://github.com/ROCm/spur/issues/452)) ([deafb10](https://github.com/ROCm/spur/commit/deafb1048a92c13d66102a125101718da955df12))
* **spurctld:** pass name filter to get_jobs in accounting reconcile ([#464](https://github.com/ROCm/spur/issues/464)) ([e970764](https://github.com/ROCm/spur/commit/e9707641f893facfe706ef0e6648d792f0655cd9))
* **spurctld:** reset stale node features/weight when NodeConfig stops matching ([#455](https://github.com/ROCm/spur/issues/455)) ([ab88dc4](https://github.com/ROCm/spur/commit/ab88dc4b4902b7c4cc14e228d519eaf27e27b50e))
* **spurctld:** retry accounting writes and reconcile drift from job store ([#428](https://github.com/ROCm/spur/issues/428)) ([c7dad32](https://github.com/ROCm/spur/commit/c7dad3249f19e538cccafc525e701bd62eea47a5))
* **spurd:** release GPU allocation on completion and post-record launch failure ([#450](https://github.com/ROCm/spur/issues/450)) ([86cd880](https://github.com/ROCm/spur/commit/86cd880067aa32c1fe3266f61de2e34747a3be9c))

## [0.5.0](https://github.com/ROCm/spur/compare/v0.4.1...v0.5.0) (2026-07-15)


### Features

* **spur-cli:** add -V/--version support across all binaries and CLI entry points ([#437](https://github.com/ROCm/spur/issues/437)) ([fa358b8](https://github.com/ROCm/spur/commit/fa358b800b0105a6aec233865ac9dffc58cc0fea))
* **spur-cli:** honor SRUN_/SALLOC_/SLURM_ env-var defaults for srun … ([#421](https://github.com/ROCm/spur/issues/421)) ([ac53a26](https://github.com/ROCm/spur/commit/ac53a26eb6f30a949f9ebef5b011a2b9236e6f55))
* **spur-cli:** support slurm-buildkite bridge (comment readback, parsable, script args, format literals, squeue --name) ([#443](https://github.com/ROCm/spur/issues/443)) ([bd4abbc](https://github.com/ROCm/spur/commit/bd4abbce80286ec67b4d0cad72644ba3dc0d017a))


### Bug Fixes

* **spur-spank:** rename SPANK exports to Slurm-compatible symbols ([#429](https://github.com/ROCm/spur/issues/429)) ([3620047](https://github.com/ROCm/spur/commit/3620047f98595ecb7c9423713089ea3ec56ebeca))
* **spurctld:** evict job to nodefail on partial multi-node dispatch failure ([#426](https://github.com/ROCm/spur/issues/426)) ([0a08108](https://github.com/ROCm/spur/commit/0a08108998b7d8408322a728f6416ef017fab0d2))
* **spurctld:** raise grpc message-size limits and cap job submission size ([#442](https://github.com/ROCm/spur/issues/442)) ([91ab393](https://github.com/ROCm/spur/commit/91ab39334ded92081b2f0a06abb200e89ff9a10a))
* **spurctld:** stop counting preemption requeues against the failure-requeue hold limit ([#431](https://github.com/ROCm/spur/issues/431)) ([35485a2](https://github.com/ROCm/spur/commit/35485a277ad95c5c0a58ab3993248d9403c6e53c))
* **spurctld:** wire QOS priority into scheduling and preemption ([#430](https://github.com/ROCm/spur/issues/430)) ([8923528](https://github.com/ROCm/spur/commit/892352891052b5c787c7eabb58f94a698e3cc677))
* **spurd:** raise RLIMIT_MEMLOCK for job steps (default unlimited) ([#449](https://github.com/ROCm/spur/issues/449)) ([d49634d](https://github.com/ROCm/spur/commit/d49634d2ddf94fc74c4273e0e3379be142febe56))

## [0.4.1](https://github.com/ROCm/spur/compare/v0.4.0...v0.4.1) (2026-07-10)


### Bug Fixes

* **spur-cli:** use connect_channel for accounting CLI commands ([#415](https://github.com/ROCm/spur/issues/415)) ([1256ea2](https://github.com/ROCm/spur/commit/1256ea2e2129a63d89f10ed281b80e4bce022e70))
* **spurd:** chown job stdout/stderr and output dirs to submitting user ([#401](https://github.com/ROCm/spur/issues/401)) ([ce7dcfd](https://github.com/ROCm/spur/commit/ce7dcfdba92971972ab294acc27d26f362c48adb))

## [0.4.0](https://github.com/ROCm/spur/compare/v0.3.0...v0.4.0) (2026-07-10)


### Features

* **accounting:** default QOS on user/account associations ([#409](https://github.com/ROCm/spur/issues/409)) ([75bf755](https://github.com/ROCm/spur/commit/75bf755683e7b55e1b81976be72c52e12904173d))
* **admission:** token-based dynamic node admission ([#318](https://github.com/ROCm/spur/issues/318)) ([91ade8a](https://github.com/ROCm/spur/commit/91ade8aad42682d30862fd0ae5bb924481fed7db))
* **config:** enforce partition allow_accounts and deny_accounts at s… ([#413](https://github.com/ROCm/spur/issues/413)) ([ba19f78](https://github.com/ROCm/spur/commit/ba19f784b9079dd67b212ff6be93a345c31eeedc))
* **e2e:** migrate K8s E2E to pytest and bundle CI test assets ([#247](https://github.com/ROCm/spur/issues/247)) ([9dc99b9](https://github.com/ROCm/spur/commit/9dc99b93df5eb03e30caf990583a5eb49797a99d))
* **e2e:** migrate native-host E2E tests from Rust to Python/pytest ([#238](https://github.com/ROCm/spur/issues/238)) ([a9ef480](https://github.com/ROCm/spur/commit/a9ef480c6f1203538b5ac71123e37f0a185d2dd5))
* **hooks:** implement Slurm-compatible prolog/epilog hook framework ([#219](https://github.com/ROCm/spur/issues/219)) ([84b7e78](https://github.com/ROCm/spur/commit/84b7e782acbcc2e9389062b9d0ff2a4c329d12fb))
* **metrics:** add controller RPC handler statistics ([#350](https://github.com/ROCm/spur/issues/350)) ([4f080f7](https://github.com/ROCm/spur/commit/4f080f7d21e5944894039ab88faea80d341feb84))
* **metrics:** add node metrics collection and /metrics/nodes export ([#237](https://github.com/ROCm/spur/issues/237)) ([7b74464](https://github.com/ROCm/spur/commit/7b744649ae5af9462ef3c34d2edf632646a82111))
* **metrics:** export job OpenMetrics from spurctld on :6822 ([#217](https://github.com/ROCm/spur/issues/217)) ([6f939d7](https://github.com/ROCm/spur/commit/6f939d722bd04a98f8c02c49869db0b5e6468d44))
* node labels and dual-path partition matching ([#276](https://github.com/ROCm/spur/issues/276)) ([7448e0e](https://github.com/ROCm/spur/commit/7448e0e988da4ffb2e8504b5be7feae8b1839a3d))
* **reservations:** persist CRUD via raft WAL and enforce scheduling parity ([#385](https://github.com/ROCm/spur/issues/385)) ([285a098](https://github.com/ROCm/spur/commit/285a0981e3b7f935226d9655428c6229538ec218))
* **scheduler:** implement event-driven scheduler wake ([#293](https://github.com/ROCm/spur/issues/293)) ([1358222](https://github.com/ROCm/spur/commit/1358222940c9df1a61bbc7453293453607ba0a7b))
* **spur-cli:** add -o/--output, -e/--error, -i/--input to srun ([#407](https://github.com/ROCm/spur/issues/407)) ([531b949](https://github.com/ROCm/spur/commit/531b949b35a824fe3387b01b964874b4f81a3a11))
* **spur-cli:** add -w/--nodelist and -x/--exclude to srun and salloc ([#380](https://github.com/ROCm/spur/issues/380)) ([ac0e4c9](https://github.com/ROCm/spur/commit/ac0e4c93c5e1f61132da81c65cf92271cdca107e))
* **spur-cli:** add sbatch --wrap for inline command submission ([#381](https://github.com/ROCm/spur/issues/381)) ([8d65736](https://github.com/ROCm/spur/commit/8d657363770bbe479f08c3cdd0525696b49de0b5))
* **spur-client:** add client-side controller failover with endpoint rotation ([#384](https://github.com/ROCm/spur/issues/384)) ([2f3b9df](https://github.com/ROCm/spur/commit/2f3b9df84c973d68517fdb5da07115c488c42079))
* **spur-cli:** read SBATCH_* env vars as sbatch flag defaults ([#374](https://github.com/ROCm/spur/issues/374)) ([c74166d](https://github.com/ROCm/spur/commit/c74166d32cbeb8100b03b72867c10c5e0bf9ec5f))
* **spur-core:** add ALL/COUNT and proto-backed JobState and NodeState conversion ([03770de](https://github.com/ROCm/spur/commit/03770de34d0cda0725a702e62e147bd46787c128))
* **spur-core:** centralize env vars with SpurEnv, add SLURM_* twins ([#311](https://github.com/ROCm/spur/issues/311)) ([64ddf60](https://github.com/ROCm/spur/commit/64ddf60dabd4bfb0b8963e95df43216e38c87694))
* **spur-devices:** CDI-based vendor-agnostic GPU device management ([#262](https://github.com/ROCm/spur/issues/262)) ([d5c819e](https://github.com/ROCm/spur/commit/d5c819e0973e2a54a886661c41abc780640cbf69))
* **spur-metrics:** add job metrics collection in spurctld ([40a3deb](https://github.com/ROCm/spur/commit/40a3deb85fbf34c57589e9cd057be87f73d1a0ac))
* **spur-metrics:** add partition, job and more metrics ([#394](https://github.com/ROCm/spur/issues/394)) ([cf126da](https://github.com/ROCm/spur/commit/cf126da3d9662e2e63c63af06be351325bc51b46))
* **spur-metrics:** add scheduler cycle and lifecycle statistics ([#359](https://github.com/ROCm/spur/issues/359)) ([a6e7cc0](https://github.com/ROCm/spur/commit/a6e7cc02d5a3d457673aa5f08ab3a02a121739c2))
* **spur:** add burst-buffer staging subsystem with capacity pool and pending reasons ([#327](https://github.com/ROCm/spur/issues/327)) ([fca5df6](https://github.com/ROCm/spur/commit/fca5df60c6cf2b4859ac26f4a58242bc6ec3ebab))
* **spur:** add Completing (CG) job state with per-node completion ([#234](https://github.com/ROCm/spur/issues/234)) ([7c0457c](https://github.com/ROCm/spur/commit/7c0457c1dcd02205da34b3d8895bbe7b42c446f6))
* **spur:** add DEADLINE job state and DeadLine reason for --deadline parity ([#263](https://github.com/ROCm/spur/issues/263)) ([0c585bd](https://github.com/ROCm/spur/commit/0c585bdc92809435de58f1b51468a3bd027c6512))
* **spur:** add job suspend/resume functionality ([#275](https://github.com/ROCm/spur/issues/275)) ([87977b1](https://github.com/ROCm/spur/commit/87977b10fb736da6a1f807bc2fb4ddd04745742e))
* **spur:** add OUT_OF_MEMORY job state from cgroup OOM detection ([#310](https://github.com/ROCm/spur/issues/310)) ([9c1e146](https://github.com/ROCm/spur/commit/9c1e1465fb6f1ffab3c2b88acb7a2e1a47478188))
* **spur:** close exit-code reporting gap — exit:signal, DerivedExitCode, reasons ([#274](https://github.com/ROCm/spur/issues/274)) ([3d89351](https://github.com/ROCm/spur/commit/3d893512318bea70c9db642b9359d079749afc36))
* **spurctld:** enforce real QoS limits via controller-side QosCache ([#325](https://github.com/ROCm/spur/issues/325)) ([672ad43](https://github.com/ROCm/spur/commit/672ad4340526780f7de08bebcc06933ff700ddcd))
* **spurctld:** serve REST API directly, remove standalone spurrestd ([#319](https://github.com/ROCm/spur/issues/319)) ([06215d2](https://github.com/ROCm/spur/commit/06215d24d83e28d7dbf54c361f3b610ad5911199))
* **spurdbd:** persist exit_signal and derived_exit_code for sacct ([#324](https://github.com/ROCm/spur/issues/324)) ([8556358](https://github.com/ROCm/spur/commit/85563588e520525c6758bd046042ad26282bf677))
* **spurdbd:** wire fairshare integration between accounting and scheduler ([8d1bcce](https://github.com/ROCm/spur/commit/8d1bcce90f4d8c02a69b4241e4f00d46dad173d9))
* **spur:** emit PartitionConfig/PartitionInactive pending reasons ([#326](https://github.com/ROCm/spur/issues/326)) ([422925d](https://github.com/ROCm/spur/commit/422925db839f61f750a7de77717917f389d4fbe4))
* **spur:** expand pending-reason vocabulary and surface reservation/license/QoS reasons ([#301](https://github.com/ROCm/spur/issues/301)) ([d5ecb3f](https://github.com/ROCm/spur/commit/d5ecb3fa13fe97afc9e3b8dd2f4f6e3d61fdf514))
* **spur:** node deregistration, drain, and job eviction ([#354](https://github.com/ROCm/spur/issues/354)) ([8fab7a9](https://github.com/ROCm/spur/commit/8fab7a9ff4333b58c97231e57507084136e5d7ef))
* use controller computed metrics in sdiag ([#332](https://github.com/ROCm/spur/issues/332)) ([5eac544](https://github.com/ROCm/spur/commit/5eac544540aa7f99e05e642358163092bb5ba45a))


### Bug Fixes

* **ci:** add node connectivity check and cross-node DNS injection ([61e632e](https://github.com/ROCm/spur/commit/61e632e362ea4ce54ef2feefd9b0163d4289f00e))
* **ci:** add node connectivity check and cross-node DNS injection ([6e03055](https://github.com/ROCm/spur/commit/6e0305537c87f3a4da3dfbe47c8e3f6175a1ef16))
* **ci:** add pre-cleanup step to native-host E2E workflow ([#330](https://github.com/ROCm/spur/issues/330)) ([cb65fda](https://github.com/ROCm/spur/commit/cb65fda76b941e117b4558a08e505f10cadc39e9))
* **ci:** add statuses permission and clean up E2E status reporting ([8b96c89](https://github.com/ROCm/spur/commit/8b96c894f7efe1e69d00161ba0b55c6d2b92b23c))
* **ci:** add statuses permission and clean up E2E status reporting ([7467851](https://github.com/ROCm/spur/commit/7467851bd51d4cb9611ad33f5a07b2bf9418c556))
* **ci:** construct image tag directly instead of relying on job outputs ([1f34859](https://github.com/ROCm/spur/commit/1f348598feaaf24d98c2c4cfd01165723af8ed90))
* **ci:** construct image tag directly instead of relying on job outputs ([5a7d032](https://github.com/ROCm/spur/commit/5a7d032260fdc5615b1dc591d399d266e8fbe4ec))
* **ci:** harden Trivy scan workflow configuration ([#334](https://github.com/ROCm/spur/issues/334)) ([f51299c](https://github.com/ROCm/spur/commit/f51299c4c6f886aa1b47ba075ffa16325d6fc985))
* **ci:** kill orphaned spur processes on BM nodes during cleanup ([e23f94f](https://github.com/ROCm/spur/commit/e23f94f264301904945d0f14974f0823db939c2b))
* **ci:** kill orphaned spur processes on BM nodes during cleanup ([c74ebdf](https://github.com/ROCm/spur/commit/c74ebdfdbef4b8d9b64dd274ddd757a03cd66c79))
* **ci:** move E2E assets to /tmp to prevent stale file contamination ([#264](https://github.com/ROCm/spur/issues/264)) ([a9ec2ce](https://github.com/ROCm/spur/commit/a9ec2ce7a694697e8da22961f6b1c661bfe48798))
* **ci:** remove stray `--` from bare-metal test binary invocations ([f879421](https://github.com/ROCm/spur/commit/f87942196d9639c281b8cb0132dbc876f0fb098a))
* **ci:** remove stray `--` from bare-metal test binary invocations ([7f8dc12](https://github.com/ROCm/spur/commit/7f8dc12dda39771865c4d34d1646d4dca3630e08))
* **ci:** use recursive glob for Dockerfile pathspecs in license checks ([#249](https://github.com/ROCm/spur/issues/249)) ([5d5bdc6](https://github.com/ROCm/spur/commit/5d5bdc63a11bce180412b7db5d01cde9f30e73cc))
* **ci:** use venv for Python E2E test deps ([#239](https://github.com/ROCm/spur/issues/239)) ([effe210](https://github.com/ROCm/spur/commit/effe2105cb8436ed024a603d755f1abfb40737b8))
* **clippy:** replace sort_by with sort_by_key using Reverse ([fff918b](https://github.com/ROCm/spur/commit/fff918bac7f175a6c5fea24b5a2e1a3d63cb5fbe))
* **deploy:** pin container tests to controller node and poll for remote inference output ([520dcf4](https://github.com/ROCm/spur/commit/520dcf49e96f68c699ed958c0b11c19057806570))
* **deploy:** pin container tests to mi300 and poll for remote inference output ([1916712](https://github.com/ROCm/spur/commit/1916712534794ddf111adf54cd452c683c4d4fd2))
* **deploy:** read log files instead of journalctl in cluster_test debug ([0e83613](https://github.com/ROCm/spur/commit/0e83613bba7dce4b2960d0ff8fec6bf60e07b637))
* **deploy:** resolve releases without the GitHub API in install.sh ([#383](https://github.com/ROCm/spur/issues/383)) ([bc3ac1c](https://github.com/ROCm/spur/commit/bc3ac1c1c876802566a4d7994394f4dc10854dfa))
* **deploy:** unpin container tests and poll for remote inference output ([ca3b3d5](https://github.com/ROCm/spur/commit/ca3b3d55009585a32b6fa5886216924322100e08))
* **deploy:** upgrade runtime base to Ubuntu 24.04 and run as non-root ([#331](https://github.com/ROCm/spur/issues/331)) ([b76a80c](https://github.com/ROCm/spur/commit/b76a80c5fa54fae6f03b5d93fdaeae94678a03f7))
* **deploy:** use correct field name for accounting host in K8s configmap ([d8511a1](https://github.com/ROCm/spur/commit/d8511a1a0b72a6d9c83f6ca7c5936579e44d285d))
* **e2e:** resolve deploy paths via SPUR_DEPLOY_DIR ([#248](https://github.com/ROCm/spur/issues/248)) ([7a80263](https://github.com/ROCm/spur/commit/7a802632a8d3afab13599c52e57fc2d887364fc6))
* **ffi:** allow not_unsafe_ptr_arg_deref on C FFI boundary functions ([ab4df8f](https://github.com/ROCm/spur/commit/ab4df8fc288c6deee0666d294718705eeed46f65))
* **install:** avoid pipefail+SIGPIPE breaking glibc version check ([#266](https://github.com/ROCm/spur/issues/266)) ([beee80b](https://github.com/ROCm/spur/commit/beee80b58f46f60f07600dc1d4944bd677233c8f))
* **raft:** fail loudly on corrupt Raft WAL/snapshot state ([#414](https://github.com/ROCm/spur/issues/414)) ([230590d](https://github.com/ROCm/spur/commit/230590d8094dfddf20f387e34f68197a6f0a3d09))
* resolve dead code and unused import warnings across 9 crates ([#155](https://github.com/ROCm/spur/issues/155)) ([1364191](https://github.com/ROCm/spur/commit/1364191344626b6ce4d1a6591f99b3f62762b289))
* resolve trivial clippy warnings across workspace ([9e1bc4e](https://github.com/ROCm/spur/commit/9e1bc4e9338554f4413b07f207619bf06e180b57))
* Resolve working dir against rootfs in non-root container fallback ([fa017a2](https://github.com/ROCm/spur/commit/fa017a2811ba2b36e9303701dee5fbc07d30fdbd))
* Resolve working dir against rootfs in non-root container fallback ([0f2c447](https://github.com/ROCm/spur/commit/0f2c44750c21b3286b15efaac6f90b45ce454738))
* **sbatch:** CLI args override #SBATCH directives ([#144](https://github.com/ROCm/spur/issues/144)) ([8c152fa](https://github.com/ROCm/spur/commit/8c152fa6b65f518c116810c04338e34a00e2d82b))
* **spur-cli:** log gRPC errors in try_stream_output polling loop ([c538602](https://github.com/ROCm/spur/commit/c538602a1adbd3430ff2a0844d1abd294dc26c43))
* **spur-cli:** parse composite keys in sshare usage display ([71f444c](https://github.com/ROCm/spur/commit/71f444c3a4d678c97e8d2a056b9283a80f9fc83b))
* **spur-cli:** replace useless vec! with array literals in sinfo tests ([383ae5e](https://github.com/ROCm/spur/commit/383ae5e12e8f34af5e4f122382c66e7b7e09f8df))
* **spur-cli:** restore SIGPIPE default to prevent broken-pipe panic ([#370](https://github.com/ROCm/spur/issues/370)) ([fdeaf3a](https://github.com/ROCm/spur/commit/fdeaf3acc0a439eb675bbc2e1f3e095ba5554511)), closes [#365](https://github.com/ROCm/spur/issues/365)
* **spur-cli:** sacct format syntax, accounting field staleness, and missing job names ([#398](https://github.com/ROCm/spur/issues/398)) ([2790200](https://github.com/ROCm/spur/commit/2790200bd7b9b07e5a3a4b4e80176529c4bbe3a8))
* **spur-cli:** use .ok() instead of manual match in sbatch script read ([ce16236](https://github.com/ROCm/spur/commit/ce1623607e575c9651c2e9ef6ff53fafe17f16e6))
* **spur-cli:** warn once on unrecognized job state in srun polling loops ([1b0aaea](https://github.com/ROCm/spur/commit/1b0aaeaa092ac5cf0e7e87863f385b27cccf19c6))
* **spur-core:** box JobSpec in WalOperation::JobSubmit to fix large_enum_variant ([c73f67e](https://github.com/ROCm/spur/commit/c73f67e5327879b896934197ce858dde38cede57))
* **spur-core:** derive Default for NodeSource instead of manual impl ([6adb88e](https://github.com/ROCm/spur/commit/6adb88e217430819e6b3ab33a43cc068d9d2e082))
* **spur-core:** serialize hook tests to avoid ETXTBSY flakiness ([#265](https://github.com/ROCm/spur/issues/265)) ([c96fa5f](https://github.com/ROCm/spur/commit/c96fa5f479fbcc5c410c29524a81bc0afcca41af))
* **spur-core:** set priority 0 for jobs submitted with hold flag ([4c6b047](https://github.com/ROCm/spur/commit/4c6b0477b5d97395acdc81d4ddc807d2cb450810))
* **spur-k8s:** handle 409 AlreadyExists in launch_job as success ([e2c3f80](https://github.com/ROCm/spur/commit/e2c3f805deba19f71e3b5ebe719179da90b32c29))
* **spur-k8s:** handle 409 AlreadyExists in launch_job as success ([0869c9e](https://github.com/ROCm/spur/commit/0869c9ef7c195c1fa7b733582ff05ad06179fbbd))
* **spur-k8s:** use shorthand field init for labels in test helper ([762a3db](https://github.com/ROCm/spur/commit/762a3db7a51c7ba4ca30d9878b35056213944d50))
* **spur-metrics:** use workspace publishable setting for the crate ([#216](https://github.com/ROCm/spur/issues/216)) ([0cd4a07](https://github.com/ROCm/spur/commit/0cd4a078fc28a7d3902d6763aa80eb5941a07e9d))
* **spur-net:** collapse consecutive str::replace calls in sanitize_name ([9077939](https://github.com/ROCm/spur/commit/9077939de3f806c748cb694e0088e76d02d96512))
* **spur-net:** replace is_some + unwrap with reference pattern matching in netrc parser ([db233d3](https://github.com/ROCm/spur/commit/db233d300b88ba0934431b3b07065f6e70bac68d))
* **spur-sched:** iterate over map values directly in het group pre-check ([9164d2c](https://github.com/ROCm/spur/commit/9164d2c2fa8c50988921eb37f8330de20ffc1463))
* **spur-tests:** remove unused imports and variables ([548d706](https://github.com/ROCm/spur/commit/548d706562e4e9dcc020a7d2149fdd27499ac34b))
* **spur-tests:** remove unused imports and variables ([374344a](https://github.com/ROCm/spur/commit/374344a26e06f4b4a1cf5138ed1e1d60cb59590f))
* **spur-tests:** replace useless vec! with array literals in test code ([d7db28a](https://github.com/ROCm/spur/commit/d7db28a842d1bea7dda106abebebf0771f386201))
* **spur-update:** simplify boolean expression in semver test ([d9feef3](https://github.com/ROCm/spur/commit/d9feef3a5e52d0b4b3356e72be2db69679764bae))
* **spurctld:** enforce job ownership on cancel, suspend, and resume ([#406](https://github.com/ROCm/spur/issues/406)) ([9958f5a](https://github.com/ROCm/spur/commit/9958f5a0802bc1d8e5280a383c3892b451846eb0))
* **spurctld:** exclusive jobs allocate full node resources ([#150](https://github.com/ROCm/spur/issues/150)) ([e38005e](https://github.com/ROCm/spur/commit/e38005eadec14129bb65e0928dd380bb9424b9aa))
* **spurctld:** free node resources on dispatch-failure requeue ([#137](https://github.com/ROCm/spur/issues/137)) ([41ffc57](https://github.com/ROCm/spur/commit/41ffc57a50e8b96f47fec053df1c64b2ef639c55))
* **spurctld:** honor node filters and match Slurm pending reasons ([#402](https://github.com/ROCm/spur/issues/402)) ([100e23b](https://github.com/ROCm/spur/commit/100e23b5a1eb742ed6c3a56be3601db5c511998f))
* **spurctld:** initialize next_job_id from first_job_id on cold start ([#371](https://github.com/ROCm/spur/issues/371)) ([ed610f4](https://github.com/ROCm/spur/commit/ed610f416af8003f71dc7a7c55207af3af23e05e))
* **spurctld:** make preemption terminate and requeue jobs correctly ([#403](https://github.com/ROCm/spur/issues/403)) ([476b40a](https://github.com/ROCm/spur/commit/476b40ae3bdcf7121372fa518d48b21996b8aff4))
* **spurctld:** persist last_purged to survive restart after log compaction ([#373](https://github.com/ROCm/spur/issues/373)) ([c2b23e9](https://github.com/ROCm/spur/commit/c2b23e966144a1b6cae612077f370a298ac71973))
* **spurctld:** remove clone on Copy types in raft vote storage ([fe83d90](https://github.com/ROCm/spur/commit/fe83d9089c9d7329d3f7a5ec77728f476bace5aa))
* **spurctld:** remove dead code and wire leader-forwarding fallback ([#156](https://github.com/ROCm/spur/issues/156)) ([c220cd8](https://github.com/ROCm/spur/commit/c220cd883c5cdbe917239e12f2797ec9758c53f0))
* **spurctld:** replace redundant pattern matching with is_err() ([e8c7921](https://github.com/ROCm/spur/commit/e8c79213f597efead1de93116c27c20e8f6d270e))
* **spurd:** Added controller info in log message ([#328](https://github.com/ROCm/spur/issues/328)) ([727af25](https://github.com/ROCm/spur/commit/727af25a34418d9e305e509f70f7c5a606a483c4))
* **spurdbd:** cast SUM aggregates to BIGINT in get_usage query ([9546ec3](https://github.com/ROCm/spur/commit/9546ec30785b39f581eabe548cfac62bb794617b))
* **spurdbd:** normalize fairshare weights and harden refresh loop ([91753d1](https://github.com/ROCm/spur/commit/91753d11212fdc3fed1813972447e1a25ba904d0))
* **spurd:** improve container init error reporting ([ca1d0ad](https://github.com/ROCm/spur/commit/ca1d0ad3c7e9c44cf0df8eda7c9ca897923cd8e1))
* **spurd:** improve container init error reporting ([d3cff08](https://github.com/ROCm/spur/commit/d3cff08fbd9ba773ee0115bfb2ea090072a7fe32))
* **spurd:** non-root container uses container's dynamic loader ([#151](https://github.com/ROCm/spur/issues/151)) ([e2f9beb](https://github.com/ROCm/spur/commit/e2f9beb54f0963a3b0d47eed4b4a678a4b703c07))
* **spurd:** prevent resource leak on force-kill cancel path ([46af47e](https://github.com/ROCm/spur/commit/46af47e138227c38f15ce68743847b5247c651a1))
* **spurd:** prevent resource leak on force-kill cancel path ([2099ea5](https://github.com/ROCm/spur/commit/2099ea5820ebcccd116a5a769e4c7e142839cb32))
* **spurd:** remove dead code and wire cgroup cleanup ([b96af21](https://github.com/ROCm/spur/commit/b96af2129cf5fb82cabeeb5894c40870247b5ba1))
* **spurd:** remove dead code and wire cgroup cleanup ([178046e](https://github.com/ROCm/spur/commit/178046eaf9b83cf8937439a82e759a01c0359163))
* **spurd:** resolve stdout/stderr paths after work_dir fallback ([#342](https://github.com/ROCm/spur/issues/342)) ([84ae6c1](https://github.com/ROCm/spur/commit/84ae6c150db54bbc91b13292a0990be78bfe1cd0))
* **spurd:** search job user's personal image store during resolution ([#138](https://github.com/ROCm/spur/issues/138)) ([75f4b53](https://github.com/ROCm/spur/commit/75f4b5353f8912ae1cfb39cc94b36ee8d11b8eb0))
* **spurd:** shell-escape argv when building job script ([#360](https://github.com/ROCm/spur/issues/360)) ([d78b1bb](https://github.com/ROCm/spur/commit/d78b1bb80a3041a2fa18cdccd4f97c6b75b639c4))
* **spurd:** use touch for file bind mounts and auto-mount DNS files i… ([#142](https://github.com/ROCm/spur/issues/142)) ([c2ce3ef](https://github.com/ROCm/spur/commit/c2ce3ef09029c841bae5f2888a62ea278b461042))
* **spur:** resolve dependency-engine gaps against array parents ([#271](https://github.com/ROCm/spur/issues/271)) ([f65a029](https://github.com/ROCm/spur/commit/f65a02939710df571a1a880ff7777af8f6e41967))
* **spur:** resolve semantic merge conflict between [#262](https://github.com/ROCm/spur/issues/262) and [#263](https://github.com/ROCm/spur/issues/263) ([#272](https://github.com/ROCm/spur/issues/272)) ([5781872](https://github.com/ROCm/spur/commit/5781872920935f38b9b9c9734bbaefb29d9a6fb3))
* **srun:** step dispatch routes to allocated node, not controller ([#153](https://github.com/ROCm/spur/issues/153)) ([bfa1c07](https://github.com/ROCm/spur/commit/bfa1c07c4b6eb3441c8e6a586f81e92d44a44cc9))


### Performance

* **deploy:** use cargo-chef to cache dependency builds in Docker ([af96542](https://github.com/ROCm/spur/commit/af9654277b68328beda03649467b3d4ee5849588))
* **spur-sched:** speed up backfill timeline queries ([#348](https://github.com/ROCm/spur/issues/348)) ([c7f56e4](https://github.com/ROCm/spur/commit/c7f56e45b0cebd8f36f4a0d955188ad00b10a07f))
* **spur-tests:** skip GPU venv provisioning when pre-existing venv available ([efdeafc](https://github.com/ROCm/spur/commit/efdeafc8db433bfc475c315059596135c2943cc3))
* **spur-tests:** skip GPU venv provisioning when pre-existing venv available ([0dc1f7a](https://github.com/ROCm/spur/commit/0dc1f7aed6178e57ba4527baa40840825f222018))
