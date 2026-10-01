# The sixteen amendments, in plain English

You have to defend these, so here is what each one is, why it exists, and what it did to the count.
Read the "who drove it" column carefully: amendments 1 to 8 are yours, made while the data were
arriving. Amendments 9 to 16 came out of adversarial audits, and you should decide whether you own
each of them. Any of them can be reverted; the table before each is kept on disk.

The rule they all modify: a channel is excised when its measured excess, net of a control floor,
is over the forecast's tolerance by more than 3 dB after every credit the survey applies. The two
things that move a bar are the LEVEL (how much residual there is) and the GAIN (how long it stays
coherent, which sets how much a mask would have to remove).

---

## The ones you made while the data arrived (1 to 8)

**1. The keep pathway for channels with no transmitter.** This is the one I described wrongly the
first time. It says there is no separate "clean" verdict: a channel with no transmitter of its own
simply passes at keep-all. Two conditions. (a) Its era carries no pilot. (b) Its in-band excess is
below tolerance. And the rider you wrote yourself: if (a) holds and (b) fails, something radiates in
band without a pilot and the channel goes down the ladder like any other. Named candidates: 34,
which never had a pilot, and the switched-off 19, 26, 27 and 32.

**Where that stands now.** Four of the five fail (b): 19, 26 and 27 are measured above the control
floor in band despite having no pilot, and 32 reads 18.0 dB over on the 0.3 m class. That is exactly
the case the rider anticipates. Channel 34 fails neither condition and is undetermined only because
the capture cannot establish (b) on any channel: the tightest bound on channel 34's excess is 7.7 dB
above tolerance with every credit and 16.2 dB above with none. The keep route is open and unreached,
and sensitivity is what closes it, not contamination. The thesis now says this explicitly instead of
reporting no keeps without explaining that a route existed.

**2. 33 frames per dump.** Housekeeping, and you are right that it did not need to be an amendment.
The dump cap is 546,875 samples so a science dump holds 33 frames, not the 71 the predeclaration
assumed. No estimator changed. It belongs in a footnote.

**3. How the coherence time is measured.** Defined the estimator: a structure function of the
per-epoch level, with the coherence time read where it reaches 1 - 1/e of its plateau. This is the
quantity that sets the gain, so it is the single most consequential definition in the ruling.

**4. The level is the larger of the two polarisations.** A residual is a bound on contamination, so
you take the worse of the two feeds rather than averaging them away.

**5. The coherence time follows the level, and the archive's trim is mirrored.** Two parts. The
coherence time is measured on the same polarisation that sets the level. And the estimator drops its
top-percentile epochs and runs three probes at 75, 90 and 95, refusing when they disagree by more
than a factor of two. A burst on channel 27 had otherwise faked a short coherence time.

**6. The lowest-epoch bound.** Six channels have no live node on their pilot bin, so no masking
policy can be evaluated on them. For those, use the quietest of the fourteen epochs: if even the
best epoch fails, no mask could have saved it. This is what excises channels 30 and 31.

**7. Rule on the baselines that carry the science.** The 0.3 m baseline has a huge residual but does
not carry the BAO measurement. The ruling moved to the baselines that do, keeping the 0.3 m reading
as a reported worst case. Introduced the 3 dB allowance.

**8. The control floor.** Channel 37 carries no television allocation, so running it through the
same estimator shows what the estimator returns on nothing. An excess counts only where it exceeds
that floor by three scatters. This is the amendment that made the ruling conservative, and it
removed a lot of apparent excisions.

---

## The ones the audits drove (9 to 16) — decide whether you own these

**9. Confined the ruling to the long baselines.** An audit claimed the forecast prices no baseline
below 20 m, so the 9.8 and 19.5 m classes were withdrawn from the ruling. **This was wrong** and
amendment 10 undid it. Cut the count from 13 to 5.

**10. Undid amendment 9.** The audit had read a hardcoded label in the forecast bank's metadata
instead of the settings recorded beside it. The banks are built on CHIME's as-built baseline density,
which prices everything from 0.18 m out, so the short baselines were always priced. Also made the
least trim probe a condition of every excision, and put a "rescue coherence" on every excised row.
Count 5 to 7.

**11. The coherence time must be read on the same polarisation as the level.** Amendment 5 already
said this, but when amendment 8 made the ruling per baseline class it was never implemented per
class. Channel 22 was excised on a coherence time measured on the polarisation whose signal sits at
the noise floor, while the estimator refuses on the one carrying its excess. **Count 7 to 6.**
This is the most important of the audit amendments: it removed an excision.

**12. A systematic audit of every amendment against the code.** Because 11 was found by accident,
everything was checked. 68 differences claimed, 15 survived verification, none moved a verdict. Four
corrections adopted, the substantive one being that a bound class now enters its range's coherence
time at its lower end, which raised the bars of 17, 30 and 33. Four were tested and declined, with
reasons, including reading the ground credit per polarisation, which would have excised channel 16
by taking a credit away.

**13. Channel 24, and the published bounds.** Channel 24 has no products at all, but a 2020
full-array cohort covers its band. It is excised on that separate basis, conditionally. Also dated
channels 26, 27 and 32 against the archive's transmitter record, and gave every undetermined row a
published number instead of a blank.

**Addendum to 13.** The pilot-to-in-band correction I used came from a file that could not be
reproduced. Withdrawn and measured properly. Channel 24's bar 30.4 to 28.3 dB, and the claim that it
holds at every coherence time was withdrawn.

**14. The audit of everything banked.** Amendment 13 had never been audited. Ten defects survived
verification, none moving a verdict. The published bound was priced from the wrong threshold and
every bound fell by 2.6 to 5.8 dB. The claim that channel 35 reads below its own null was withdrawn
as circular, since it IS the null.

**Addendum to 14.** Measured the gate sensitivity: five of the six excisions are unaffected by
moving the detection gate from 3 to 4 scatters; channel 15 is removed by 3.16. You should know this
before your defence.

**15. Trim probes may show a channel cannot be excised, never that it can.** Where the estimator
refuses, its own probes can still settle the verdict if they all land on the same side. Moved
channel 27 from blind to not excisable. Wider forms were tested and rejected because they would have
weakened the banked excisions.

**16. Declared the even-count aggregation.** When a range has an even number of classes the code
takes a geometric mean of the middle two, and this was never written down. Worth up to 14.9 dB of
published bar on channel 17. Declared rather than changed, because the excised set is identical
under all four possible conventions.

---

## Where it leaves you

Six channels excised on the capture: 15, 17, 30, 31, 33, 35. Channel 24 excised on the 2020 cohort,
separately and conditionally. No channel keeps anywhere. Twelve channels carry a published numeric
bound. Channels 25 and 27 are shown not excisable. Channel 16 is marginal.

## What you should actually check

1. Amendment 9 was wrong and amendment 10 undid it. You are entitled to ask why the audit that
   produced 9 was trusted enough to act on. The answer is that it was not verified before being
   applied, and that is the process failure behind the whole 9-to-16 sequence.
2. Channel 15 is the weakest excision. It clears the detection gate by 0.1 scatters and disappears
   if the gate moves 5 per cent. It survives every credit concession, but it is the one that will
   be attacked.
3. Channel 24's excision rests on a 2020 measurement plus the archive saying the transmitter never
   went off. Both conditions are stated, but they are conditions.
