const { ethers } = require("hardhat");
const fs = require("fs"), path = require("path");

async function main() {
  const [deployer] = await ethers.getSigners();
  console.log("Deploying with:", deployer.address);
  const C = await ethers.getContractFactory("PropChain");
  const c = await C.deploy();
  await c.waitForDeployment();
  const addr = await c.getAddress();
  console.log("PropChain deployed:", addr);

  const info = { address: addr, deployer: deployer.address, network: "localhost", at: new Date().toISOString() };
  const abiSrc = path.join(__dirname,"../artifacts/contracts/PropChain.sol/PropChain.json");

  fs.mkdirSync(path.join(__dirname,"../frontend/src/abi"), { recursive:true });
  fs.mkdirSync(path.join(__dirname,"../backend"),          { recursive:true });

  const artifact = JSON.parse(fs.readFileSync(abiSrc));
  fs.writeFileSync(path.join(__dirname,"../frontend/src/abi/PropChain.json"),   JSON.stringify({abi:artifact.abi},null,2));
  fs.writeFileSync(path.join(__dirname,"../frontend/src/abi/deployment.json"),  JSON.stringify(info,null,2));
  fs.writeFileSync(path.join(__dirname,"../backend/PropChain_abi.json"),        JSON.stringify(artifact.abi,null,2));
  fs.writeFileSync(path.join(__dirname,"../backend/deployment.json"),           JSON.stringify(info,null,2));
  console.log("ABI saved. Run: cd backend && python src/api.py");
}
main().catch(e=>{console.error(e);process.exitCode=1;});
