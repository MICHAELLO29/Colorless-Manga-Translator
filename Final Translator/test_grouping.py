import dataclasses

@dataclasses.dataclass
class TextBlock:
    box: tuple
    text: str

blocks = [
    TextBlock((294, 22, 19, 89), "Knight's Order insurance is..."),
    TextBlock((272, 22, 19, 100), "It's a lifetime contract, though~"),
    TextBlock((96, 57, 23, 109), "Comms jammer"),
    TextBlock((74, 55, 21, 123), "This contract plan is..."),
    TextBlock((421, 169, 14, 66), "Hah?!"),
    TextBlock((386, 64, 37, 313), "He's weirdly knowledgeable about contracts."),
    TextBlock((344, 61, 38, 216), "Deputy Commander Nacht"),
    TextBlock((303, 167, 20, 95), "About the rental agreement..."),
    TextBlock((284, 167, 17, 109), "Grimoire's"),
    TextBlock((264, 168, 19, 67), "Just gotta copy this..."),
    TextBlock((115, 265, 19, 51), "Lately..."),
    TextBlock((94, 263, 19, 101), "Used my magic"),
    TextBlock((72, 263, 19, 100), "The contract's signed and sealed...")
]

def group_blocks_new(blocks):
    used = set()
    groups = []
    for i, block_a in enumerate(blocks):
        if i in used: continue
        group = [block_a]
        used.add(i)
        changed = True
        while changed:
            changed = False
            for j, block_b in enumerate(blocks):
                if j in used: continue
                xb, yb, wb, hb = block_b.box
                is_close = False
                for block_in_group in group:
                    xa, ya, wa, ha = block_in_group.box
                    
                    # 1. Vertical overlap check
                    y_overlap = min(ya + ha, yb + hb) - max(ya, yb)
                    if y_overlap < min(ha, hb) * 0.25:
                        continue
                        
                    # 2. Similar width check (same font size)
                    if max(wa, wb) > min(wa, wb) * 1.8:
                        continue
                        
                    # 3. Horizontal proximity check
                    x_gap = max(0, max(xa, xb) - min(xa + wa, xb + wb))
                    if x_gap > min(wa, wb) * 1.2:
                        continue
                        
                    # 4. Top-alignment check (columns in same bubble usually start near each other)
                    # Allow some variance, e.g. 50% of the maximum height
                    if abs(ya - yb) > max(ha, hb) * 0.5:
                        continue
                        
                    is_close = True
                    break
                        
                if is_close:
                    group.append(block_b)
                    used.add(j)
                    changed = True
        groups.append(group)
    return groups

print("NEW GROUPS:")
for g in group_blocks_new(blocks):
    print([b.text for b in g])
