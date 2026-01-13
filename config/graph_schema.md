# Triple_RAG Neo4j Graph Database Schema

### Primary Node Types
- **Product** (1,933 nodes): Product-related nodes, including metals, minerals, chemical products, etc.
- **Organization** (1,236 nodes): Organizations, institutions, associations, etc.
- **Location** (1,200 nodes): Geographic locations, including countries, regions, cities, etc.
- **Company** (1,149 nodes): Companies and enterprises, including mining, manufacturing, and trading companies
- **other** (384 nodes): Other uncategorized nodes
- **Other** (96 nodes): Other classified nodes
- **Person** (69 nodes): Individuals, including analysts, executives, experts, etc.
- **Event** (47 nodes): Events, including conferences, policy changes, market events, etc.

## Relationship Types (Sorted by Count)

### Major Relationship Types (>100 relationships)
1. **located_in** (1,652 relationships): Location relationship, indicates an entity is located in a geographic location
2. **supplies** (1,221 relationships): Supply relationship, indicates supplying products or materials
3. **produces** (1,091 relationships): Production relationship, indicates producing a certain product
4. **other** (1,006 relationships): Other uncategorized relationships
5. **operates** (608 relationships): Operation relationship, indicates operating a facility or business
6. **owns** (512 relationships): Ownership relationship, indicates owning an entity or asset
7. **purchases** (326 relationships): Purchase relationship, indicates purchasing a product or service

### Medium Relationship Types (10-100 relationships)
- **reports** (85 relationships): Reporting relationship
- **assesses** (37 relationships): Assessment relationship
- **analyzes** (35 relationships): Analysis relationship
- **affects** (34 relationships): Impact relationship
- **related_to** (31 relationships): Related relationship
- **trades** (27 relationships): Trading relationship
- **has** (23 relationships): Possession relationship
- **imports** (22 relationships): Import relationship
- **employs** (20 relationships): Employment relationship
- **works_for** (18 relationships): Work relationship
- **exports** (17 relationships): Export relationship
- **predicts** (16 relationships): Prediction relationship
- **said** (15 relationships): Statement relationship
- **assessed** (14 relationships): Assessment relationship
- **estimates** (13 relationships): Estimation relationship
- **invests_in** (12 relationships): Investment relationship
- **used_in** (11 relationships): Usage relationship

### Supply Chain Characteristics
- Covers complete supply chain of metals, minerals, and chemical products
- Includes production relationships from raw materials to final products
- Involves major global production and consumption regions
