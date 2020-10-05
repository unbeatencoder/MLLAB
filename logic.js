/*
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

'use strict';
/**
 * Write your transction processor functions here
 */

/**
 * AggregateGradientsFunction
 * @param {org.example.mynetwork.AggregateGradientsFunction} AggregateGradientsFunction
 * @transaction
 */
async function aggregateGradientsTransactionFunction(AggregateGradientsFunction) {
    // Save the old value of the asset.
    //const GradientPerIterationValue = AggregateGradientsFunction.gradientPerIterationValues;

    var sum = 0;
    //var i = 0;
    //for(var i = 0;i<GradientPerIterationValue.length; i++){
    //    sum = sum + GradientPerIterationValue[i].gradientvalues[0];
    //}
    //console.log(sum);
	  var a = 10;
  	var b = 20;
  	var numBits = 256;
	  var keys = paillier.generateKeys(numBits);
  	var publickey = keys.pub.n.toString();
  	var privkey = keys.sec.lambda.toString();
	  var encA = keys.pub.encrypt(nbv(a));
    var encB = keys.pub.encrypt(nbv(b));
    var encAB = keys.pub.add(encA,encB);
    var plaintext = keys.sec.decrypt(encAB).toString(10);
    console.log(plaintext);
     // Get the asset registry for the asset.
    sum = sum + plaintext;
    return getAssetRegistry('org.example.mynetwork.AggregateGradientsOutput')
    .then(function(result){
  	var factory=getFactory();
  	var random=Math.floor((Math.random() * 1000) + 1);
    var aggregateGradientsOutputID= random;  
    var new_aggregateGradientsObject =  factory.newResource('org.example.mynetwork', 'AggregateGradientsOutput', 'AGG_20');
    // Update the asset in the asset registry.
    var stringsum = parseInt(sum);
    new_aggregateGradientsObject.aggregateValues = [stringsum];
    return result.add(new_aggregateGradientsObject);
   });
}
