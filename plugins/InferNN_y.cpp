/*
 * Example plugin to demonstrate the direct multi-threaded inference on CNN with TensorFlow 2.
 */

#include <memory>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cmath>
#include <cfloat>
#include <iostream>
#include <cstring>
#include <algorithm>
#include <vector>
#include <iomanip>
#include <sstream>
#include <fstream>

#include <TF1.h>
#include "Math/MinimizerOptions.h"
#include <TCanvas.h>
#include <TGraphErrors.h>
#include <TMath.h>
#include "TROOT.h"
#include "TSystem.h"
#include "TFile.h"
#include "TObject.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TProfile.h"
#include "TStyle.h"
#include "TF1.h"
#include "TCanvas.h"
#include "TPostScript.h"
#include "Math/DistFunc.h"
#include "TTree.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/stream/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "PhysicsTools/TensorFlow/interface/TensorFlow.h"

#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DQM/SiPixelPhase1Common/interface/SiPixelPhase1Base.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "RecoLocalTracker/SiPixelRecHits/interface/SiPixelTemplateReco.h"
#include "CalibTracker/Records/interface/SiPixelTemplateDBObjectESProducerRcd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/PixelGeomDetUnit.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "Geometry/CommonTopologies/interface/Topology.h"

#include "SimDataFormats/TrackingHit/interface/PSimHit.h"  
#include "SimTracker/TrackerHitAssociation/interface/TrackerHitAssociator.h" 
#include "DataFormats/SiPixelDigi/interface/PixelDigi.h"
#include "SimDataFormats/TrackerDigiSimLink/interface/PixelDigiSimLink.h"


#include "TrackingTools/TrackAssociator/interface/TrackDetectorAssociator.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"
#include "TrackingTools/TrackAssociator/interface/TrackAssociatorParameters.h"
//#include "TrackingTools/TrackAssociator/interface/TrackDetectorAssociator.h"


class TObject;
class TTree;
class TH1D;
class TFile;

using namespace std;
using namespace edm;
using namespace reco;
// define the cache object
// it could handle graph loading and destruction on its own,
// but in this example, we define it as a logicless container
struct CacheData {
	CacheData() : graphDef(nullptr) {}
	std::atomic<tensorflow::GraphDef*> graphDef;
};

class InferNN_y : public edm::stream::EDAnalyzer<edm::GlobalCache<CacheData>> {
public:
	explicit InferNN_y(const edm::ParameterSet&, const CacheData*);
	~InferNN_y(){};

	static void fillDescriptions(edm::ConfigurationDescriptions&);

	// two additional static methods for handling the global cache
	static std::unique_ptr<CacheData> initializeGlobalCache(const edm::ParameterSet&);
	static void globalEndJob(const CacheData*); // does it have to be static

private:
	void beginJob();
	void analyze(const edm::Event&, const edm::EventSetup&);
	void endJob();

	std::string inputTensorName_y, inputTensorName_y, anglesTensorName_y, anglesTensorName_y;
	std::string outputTensorName_;
	//std::string     fRootFileName;
	tensorflow::Session* session_y;
	bool use_det_angles;
	std::string cpe;
	TFile *fFile; TTree *fTree;
	static const int MAXCLUSTER = 100000;
	static const int SIMHITPERCLMAX = 10;             // max number of simhits associated with a cluster/rechit
	float fClSimHitLx[MAXCLUSTER][SIMHITPERCLMAX];    // X local position of simhit 
	float fClSimHitLy[MAXCLUSTER][SIMHITPERCLMAX];
	float y_gen[MAXCLUSTER], y_nn[MAXCLUSTER]; 
	int count; char path[100], infile1[300], infile2[300], infile3[300], infile4[300];
	edm::InputTag fTrackCollectionLabel, fPrimaryVertexCollectionLabel;
	
	edm::EDGetTokenT<std::vector<reco::Track>> TrackToken;
	edm::EDGetTokenT<reco::VertexCollection> VertexCollectionToken;
	edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> TrackerTopoToken;

	edm::EDGetTokenT<edm::DetSetVector<PixelDigiSimLink>> PixelDigiSimLinkToken;
	edm::EDGetTokenT<edm::SimTrackContainer> SimTrackContainerToken;
	edm::EDGetTokenT<edm::SimVertexContainer> SimVertexContainerToken;

	FILE *nn_file, *gen_file, *sim_file, *clustersize_y_file;
	TrackerHitAssociator::Config trackerHitAssociatorConfig_;
	float micronsToCm = 1e-4;
	float pixelsize_x = 100., pixelsize_y = 150., pixelsize_z = 285.0;
	int mid_x = 0, mid_y = 0;
	float clsize_1[MAXCLUSTER][2], clsize_2[MAXCLUSTER][2], clsize_3[MAXCLUSTER][2], clsize_4[MAXCLUSTER][2], clsize_5[MAXCLUSTER][2], clsize_6[MAXCLUSTER][2];
	

	};

	std::unique_ptr<CacheData> InferNN_y::initializeGlobalCache(const edm::ParameterSet& config) 
	{

	// this method is supposed to create, initialize and return a CacheData instance
		CacheData* cacheData = new CacheData();

	// load the graph def and save it
		std::string graphPath_y = config.getParameter<std::string>("graphPath_y");
		cacheData->graphDef = tensorflow::loadGraphDef(graphPath_y);

	// set tensorflow log leven to warning
		tensorflow::setLogging("2");
	//init();

		return std::unique_ptr<CacheData>(cacheData);
	}

	void InferNN_y::globalEndJob(const CacheData* cacheData) {
	// reset the graphDef
		if (cacheData->graphDef != nullptr) {
			delete cacheData->graphDef;
		}

	}

	void InferNN_y::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
	// defining this function will lead to a *_cfi file being generated when compiling
		edm::ParameterSetDescription desc;
		desc.add<std::string>("graphPath_y");
		desc.add<std::string>("inputTensorName_y");
		desc.add<std::string>("anglesTensorName_y");
		desc.add<std::string>("outputTensorName");
		desc.add<bool>("use_det_angles");
		desc.add<std::string>("cpe");
		desc.add<bool>("associatePixel");
		desc.add<bool>("associateStrip");
		desc.add<bool>("associateRecoTracks");
		desc.add<std::string>("pixelSimLinkSrc");
		desc.add<std::string>("stripSimLinkSrc");
		desc.add<std::vector<std::string>>("ROUList");
		descriptions.addWithDefaultLabel(desc);
	}

	InferNN_y::InferNN_y(const edm::ParameterSet& config, const CacheData* cacheData)
	: inputTensorName_y(config.getParameter<std::string>("inputTensorName_y")),
	anglesTensorName_y(config.getParameter<std::string>("anglesTensorName_y")),
	outputTensorName_(config.getParameter<std::string>("outputTensorName")),
	session_y(tensorflow::createSession(cacheData->graphDef)),
	use_det_angles(config.getParameter<bool>("use_det_angles")),
	cpe(config.getParameter<std::string>("cpe")),
	fTrackCollectionLabel(config.getUntrackedParameter<InputTag>("trackCollectionLabel", edm::InputTag("generalTracks"))),
	fPrimaryVertexCollectionLabel(config.getUntrackedParameter<InputTag>("PrimaryVertexCollectionLabel", edm::InputTag("offlinePrimaryVertices"))),
	trackerHitAssociatorConfig_(config, consumesCollector()) {

		TrackToken              = consumes <std::vector<reco::Track>>(fTrackCollectionLabel) ;
		VertexCollectionToken   = consumes <reco::VertexCollection>(fPrimaryVertexCollectionLabel) ;
		TrackerTopoToken        = esConsumes <TrackerTopology, TrackerTopologyRcd>();

		PixelDigiSimLinkToken   = consumes <edm::DetSetVector<PixelDigiSimLink>>(edm::InputTag("simSiPixelDigis")); 
		SimTrackContainerToken  = consumes <edm::SimTrackContainer>(edm::InputTag("g4SimHits")); 
		SimVertexContainerToken = consumes <edm::SimVertexContainer>(edm::InputTag("g4SimHits")); 
		count = 0;

	//initializations
		for(int i=0;i<MAXCLUSTER;i++){
			y_nn[i]=-999.0;
			y_gen[i]=-999.0;
			
			for(int j=0;j<SIMHITPERCLMAX;j++){
				fClSimHitLx[i][j]=-999.0;
				fClSimHitLy[i][j]=-999.0;
			}
			for(int j=0;j<2;j++){
				clsize_1[i][j]=-999.0;
				clsize_2[i][j]=-999.0;
				clsize_3[i][j]=-999.0;
				clsize_4[i][j]=-999.0;			
				clsize_5[i][j]=-999.0;
				clsize_6[i][j]=-999.0;
			}
			
		}
		sprintf(path,"TrackerStuff/PixelHitsCNN/txt_files");

		sprintf(infile1,"generic_MC_x.txt");
	//	gen_file = fopen(infile1, "w");

		sprintf(infile2,"%s/simhits_MC.txt",path);
		sim_file = fopen(infile2, "w");

		sprintf(infile3,"%s/%s_MC_x.txt",path,cpe);
		nn_file = fopen(infile3, "w");

		sprintf(infile4,"%s/%s_MC_perclustersize_x.txt",path,cpe);
		clustersize_y_file = fopen(infile4, "w");

		
	}

	void InferNN_y::beginJob() {

	}

	void InferNN_y::endJob() {
	// close the session
		tensorflow::closeSession(session_y);

		fclose(nn_file);
		fclose(sim_file);

	}

	void InferNN_y::analyze(const edm::Event& event, const edm::EventSetup& setup) {


		if (sim_file==NULL) {
			printf("couldn't open simhit output file/n");
			return ;
		}
	
		if (nn_file==NULL) {
			printf("couldn't open residual output file/n");
			return ;
		}
		edm::ESHandle<TrackerTopology> tTopoHandle = setup.getHandle(TrackerTopoToken);
		auto const& tkTpl = *tTopoHandle;
		// get geometry
	
		std::vector<PSimHit> vec_simhits_assoc;
		TrackerHitAssociator *associate(0);

		associate = new TrackerHitAssociator(event,trackerHitAssociatorConfig_);

	//get the map
		edm::Handle<reco::TrackCollection> tracks;

		try {
			event.getByToken(TrackToken, tracks);
		}catch (cms::Exception &ex) {
	//if (fVerbose > 1) 
			cout << "No Track collection with label " << fTrackCollectionLabel << endl;
		}
		if (tracks.isValid()) {
			const std::vector<reco::Track> trackColl = *(tracks.product());
		//nTk = trackColl.size();
		//if (fVerbose > 1) 
		//cout << "--> Track collection size: " << nTk << endl;
		} else {
  	//if (fVerbose > 1)
			cout << "--> No valid track collection" << endl;
		}
		if (!tracks.isValid()) {
			cout << "track collection is not valid" <<endl;
			return;
		}

		float clusbuf[TXSIZE][TYSIZE];

		static int ix,iy;
		int prev_count = count;
		for (auto const& track : *tracks) {

			auto etatk = track.eta();

			auto const& trajParams = track.extra()->trajParams();
			assert(trajParams.size() == track.recHitsSize());
			auto hb = track.recHitsBegin();

			for (unsigned int h = 0; h < track.recHitsSize(); h++) {
				auto hit = *(hb + h);
				if (!hit->isValid())
					continue;
				auto id = hit->geographicalId();

			// check that we are in the pixel detector
				auto subdetid = (id.subdetId());

				if (subdetid != PixelSubdetector::PixelBarrel) //&& subdetid != PixelSubdetector::PixelEndcap)
					continue;
			if (tkTpl.pxbLayer(id) != 1) //only L1
				continue;
			

			auto pixhit = dynamic_cast<const SiPixelRecHit*>(hit->hit());
			if (!pixhit)
				continue;


			//some initialization
			for(int j=0; j<TXSIZE; ++j) {for(int i=0; i<TYSIZE; ++i) {clusbuf[j][i] = 0.;} } 


			// get the cluster
				auto clustp = pixhit->cluster();
			if (clustp.isNull())
				continue;
			auto const& cluster = *clustp;
			const std::vector<SiPixelCluster::Pixel> pixelsVec = cluster.pixels();

			auto const& ltp = trajParams[h];

			 // Preparing to retrieve ADC counts from the SiPixeltheClusterParam.theCluster->  In the cluster,
				  // we have the following:
				  //   int minPixelRow(); // Minimum pixel index in the x direction (low edge).
				  //   int maxPixelRow(); // Maximum pixel index in the x direction (top edge).
				  //   int minPixelCol(); // Minimum pixel index in the y direction (left edge).
				  //   int maxPixelCol(); // Maximum pixel index in the y direction (right edge).
				  // So the pixels from minPixelRow() will go into clust_array_2d[0][*],
				  // and the pixels from minPixelCol() will go into clust_array_2d[*][0].


			int row_offset = cluster.minPixelRow();
			int col_offset = cluster.minPixelCol();
	         //printf("cluster.minPixelRow() = %i\n",cluster.minPixelRow());
	         //printf("cluster.minPixelCol() = %i\n",cluster.minPixelCol());
			// Store the coordinates of the center of the (0,0) pixel of the array that
			// gets passed to PixelTempReco1D
			// Will add these values to the output of  PixelTempReco1D
			float tmp_x = float(row_offset) + 0.5f;
			float tmp_y = float(col_offset) + 0.5f;

			float cotAlpha=ltp.dxdz();
			float cotBeta=ltp.dydz();
			//https://github.com/cms-sw/cmssw/blob/master/RecoLocalTracker/SiPixelRecHits/src/PixelCPEBase.cc#L263-L272
			LocalPoint trk_lp = ltp.position();
			float trk_lp_x = trk_lp.x();
			float trk_lp_y = trk_lp.y();

			Topology::LocalTrackPred loc_trk_pred =Topology::LocalTrackPred(trk_lp_x, trk_lp_y, cotAlpha, cotBeta);
			LocalPoint lp; 
			auto geomdetunit = dynamic_cast<const PixelGeomDetUnit*>(pixhit->detUnit());
			auto const& topol = geomdetunit->specificTopology();
			lp = topol.localPosition(MeasurementPoint(tmp_x, tmp_y), loc_trk_pred);
			if(use_det_angles) lp = topol.localPosition(MeasurementPoint(tmp_x, tmp_y));


			if(use_det_angles){
				auto const& theOrigin = geomdetunit->surface().toLocal(GlobalPoint(0, 0, 0));
				LocalPoint lp2 = topol.localPosition(
					MeasurementPoint(cluster.x(), cluster.y()));
				auto gvx = lp2.x() - theOrigin.x();
				auto gvy = lp2.y() - theOrigin.y();
				auto gvz = -1.f /theOrigin.z();	
					// calculate angles
				cotAlpha = gvx * gvz;
				cotBeta = gvy * gvz;
			}

				  // first compute matrix size
			int mrow = 0, mcol = 0;
			for (int i = 0; i != cluster.size(); ++i) {
				auto pix = cluster.pixel(i);
				int irow = int(pix.x);
				int icol = int(pix.y);
				mrow = std::max(mrow, irow);
				mcol = std::max(mcol, icol);
			}
			mrow -= row_offset;
			mrow += 1;
			mrow = std::min(mrow, TXSIZE);
			mcol -= col_offset;
			mcol += 1;
			mcol = std::min(mcol, TYSIZE);
			assert(mrow > 0);
			assert(mcol > 0);

			bool bigPixel=false;
			int irow_sum = 0, icol_sum = 0;
			for (int i = 0; i < cluster.size(); ++i) {
				auto pix = cluster.pixel(i);
				int irow = int(pix.x) - row_offset;
				int icol = int(pix.y) - col_offset;
					//double pixels skip
				if ((int)pix.x == 79 || (int)pix.x == 80){
					bigPixel=true; break;
				}
				if ((int)pix.y % 52 == 0 || (int)pix.y % 52 == 51 ){
					bigPixel=true; break;
				}
				irow_sum+=irow;
				icol_sum+=icol;

			}
			if(bigPixel) continue;
			int clustersize_x = cluster.sizeX(), clustersize_y = cluster.sizeY();
			mid_x = round(float(irow_sum)/float(cluster.size()));
			mid_y = round(float(icol_sum)/float(cluster.size()));
			int offset_x = 6 - mid_x;
			int offset_y = 10 - mid_y;


  // Copy clust's pixels (calibrated in electrons) into clusMatrix;
			for (int i = 0; i < cluster.size(); ++i) {
				auto pix = cluster.pixel(i);
				int irow = int(pix.x) - row_offset + offset_x;
				int icol = int(pix.y) - col_offset + offset_y;
					//printf("irow = %i, icol = %i\n",irow,icol);
					//printf("mrow = %i, mcol = %i\n",mrow,mcol);

				if ((irow > mrow+offset_x) || (icol > mcol+offset_y)) continue;
				clusbuf[irow][icol] = float(pix.adc);
 				    //printf("pix[%i].adc = %i, pix.x = %i, pix.y = %i, irow = %i, icol = %i\n",i,pix.adc,pix.x,pix.y,(int(pix.x) - row_offset),int(pix.y) - col_offset);

			}

				//===============================
				// define a tensor and fill it with cluster projection
			tensorflow::Tensor cluster_flat_y(tensorflow::DT_FLOAT, {1,TXSIZE,1});
			tensorflow::Tensor cluster_(tensorflow::DT_FLOAT, {1,TXSIZE,TYSIZE,1});
    		// angles
			tensorflow::Tensor angles(tensorflow::DT_FLOAT, {1,2});
			angles.tensor<float,2>()(0, 0) = cotAlpha;
			angles.tensor<float,2>()(0, 1) = cotBeta;

			for (size_t i = 0; i < TXSIZE; i++) {
				cluster_flat_y.tensor<float,3>()(0, i, 0) = 0;
				for (size_t j = 0; j < TYSIZE; j++){
            //1D projection in x
					cluster_flat_y.tensor<float,3>()(0, i, 0) += clusbuf[i][j];
					cluster_.tensor<float,4>()(0, i, j, 0) = clusbuf[i][j];
					
					//printf("%i ",int(clusbuf[i][j]));

				}
					//printf("\n");
			}
			
				// define the output and run
			std::vector<tensorflow::Tensor> output_y;
			if(cpe=="cnn2d") tensorflow::run(session_y, {{inputTensorName_y,cluster_}, {anglesTensorName_y,angles}}, {outputTensorName_}, &output_y);
			else tensorflow::run(session_y, {{inputTensorName_y,cluster_flat_y}, {anglesTensorName_y,angles}}, {outputTensorName_}, &output_y);
				// convert microns to cms
			y_nn[count] = output_y[0].matrix<float>()(0,0);
				//printf("x = %f\n",y_nn[count]);

			y_nn[count] = (y_nn[count]+pixelsize_x*(mid_x))*micronsToCm; 

			//	printf("cota = %f, cotb = %f, y_nn = %f\n",cotAlpha,cotBeta,y_nn[count]);
				// go back to module coordinate system
			y_nn[count]+=lp.x(); 
				// get the generic position
			y_gen[count] = hit->localPosition().x();
				//get sim hits
			vec_simhits_assoc.clear();
			vec_simhits_assoc = associate->associateHit(*pixhit);

			int iSimHit = 0;

			for (std::vector<PSimHit>::const_iterator m = vec_simhits_assoc.begin(); 
				m < vec_simhits_assoc.end() && iSimHit < SIMHITPERCLMAX; ++m) 
			{



				fClSimHitLx[count][iSimHit]    = ( m->entryPoint().x() + m->exitPoint().x() ) / 2.0;
				fClSimHitLy[count][iSimHit]    = ( m->entryPoint().y() + m->exitPoint().y() ) / 2.0;

				++iSimHit;

            } // end sim hit loop
				
		//	printf("Generic position: %f\n ",(y_gen[count]-lp.x())*1e4);
		//	printf("nn position: %f\n ",(y_nn[count]-lp.x())*1e4);
		//	printf("simhit_x =");
	
//			printf("%i\n",count);
            switch(clustersize_x){
            	case 1: 
            	clsize_1[count][0]=y_nn[count];
            	break;
            	case 2: 
            	clsize_2[count][0]=y_nn[count];
            	break;
            	case 3: 
            	clsize_3[count][0]=y_nn[count];
            	break;
            	case 4: 
            	clsize_4[count][0]=y_nn[count];
            	break;
            	case 5: 
            	clsize_5[count][0]=y_nn[count];
            	break;
            	case 6: 
            	clsize_6[count][0]=y_nn[count];
            	break;
            }
            count++;

        }
    }


    for(int i=prev_count;i<count;i++){
    	for(int j=0; j<SIMHITPERCLMAX;j++){
    		fprintf(sim_file,"%f ", fClSimHitLx[i][j]);
    	}
    	for(int j=0; j<SIMHITPERCLMAX;j++){
    		fprintf(sim_file,"%f ", fClSimHitLy[i][j]);
    	}
    	fprintf(sim_file,"\n");
    	fprintf(nn_file,"%f %f\n", y_gen[i],y_nn[i]);

    	fprintf(clustersize_y_file,"%f %f %f %f %f %f\n", clsize_1[i][0],clsize_2[i][0],clsize_3[i][0],clsize_4[i][0],clsize_5[i][0],clsize_6[i][0]);
    }
    

}
DEFINE_FWK_MODULE(InferNN_y);