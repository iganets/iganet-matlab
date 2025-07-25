/**
   @file mexiganet/src/mexNeumannPoisson.cpp

   @brief IgANets for solving the Poisson problem with homogeneous Neumann boundary conditions

   @author Matthias Moller

   @copyright This file is part of the IgANet project

   This Source Code Form is subject to the terms of the Mozilla Public
   License, v. 2.0. If a copy of the MPL was not distributed with this
   file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#include <codecvt>  // For std::codecvt_utf8_utf16
#include <locale>   // For std::wstring_convert

#include <iganet.h>
#include <mexAdapter.hpp>

using namespace iganet::literals;

namespace iganet { namespace mex {
    
/// @brief Specialization of the abstract IgANet class for Poisson's equation
template <typename Optimizer, typename GeometryMap, typename Variable>
class Poisson : public iganet::IgANet<Optimizer, GeometryMap, Variable>,
                public iganet::IgANetCustomizable<GeometryMap, Variable> {

private:
  /// @brief Type of the base class
  using Base = iganet::IgANet<Optimizer, GeometryMap, Variable>;

  /// @brief Collocation points
  typename Base::variable_collPts_type collPts_;

  /// @brief Reference solution
  Variable ref_;

  /// @brief Type of the customizable class
  using Customizable = iganet::IgANetCustomizable<GeometryMap, Variable>;

  /// @brief Knot indices of variables
  typename Customizable::variable_interior_knot_indices_type var_knot_indices_;

  /// @brief Coefficient indices of variables
  typename Customizable::variable_interior_coeff_indices_type
      var_coeff_indices_;

  /// @brief Knot indices of the geometry map
  typename Customizable::geometryMap_interior_knot_indices_type G_knot_indices_;

  /// @brief Coefficient indices of the geometry map
  typename Customizable::geometryMap_interior_coeff_indices_type
      G_coeff_indices_;

public:
  /// @brief Constructor
  template <std::size_t GeometryMapNumCoeffs, std::size_t VariableNumCoeffs>
  Poisson(const std::vector<int64_t> &layers,
          const std::vector<std::vector<std::any>> &activations,
          const std::array<int64_t, GeometryMapNumCoeffs> &geometryMapNumCoeffs,
          const std::array<int64_t, VariableNumCoeffs> &variableNumCoeffs)
      : Base(layers,
             activations,
             geometryMapNumCoeffs,
             variableNumCoeffs),
        ref_(variableNumCoeffs) {}

  /// @brief Returns a constant reference to the collocation points
  auto const &collPts() const { return collPts_; }

  /// @brief Returns a constant reference to the reference solution
  auto const &ref() const { return ref_; }

  /// @brief Returns a non-constant reference to the reference solution
  auto &ref() { return ref_; }

  /// @brief Initializes the epoch
  ///
  /// @param[in] epoch Epoch number
  bool epoch(int64_t epoch) override {
    // In the very first epoch we need to generate the sampling points
    // for the inputs and the sampling points in the function space of
    // the variables since otherwise the respective tensors would be
    // empty. In all further epochs no updates are needed since we do
    // not change the inputs nor the variable function space.
    if (epoch == 0) {
      Base::inputs(epoch);
      collPts_ = Base::variable_collPts(iganet::collPts::greville_ref1);

      var_knot_indices_ =
          Base::f_.template find_knot_indices<iganet::functionspace::interior>(
              collPts_.first);
      var_coeff_indices_ =
          Base::f_.template find_coeff_indices<iganet::functionspace::interior>(
              var_knot_indices_);

      G_knot_indices_ =
          Base::G_.template find_knot_indices<iganet::functionspace::interior>(
              collPts_.first);
      G_coeff_indices_ =
          Base::G_.template find_coeff_indices<iganet::functionspace::interior>(
              G_knot_indices_);

      return true;
    } else
      return false;
  }

  /// @brief Computes the loss function
  ///
  /// @param[in] outputs Output of the network
  ///
  /// @param[in] epoch Epoch number
  torch::Tensor loss(const torch::Tensor &outputs, int64_t epoch) override {

    // Cast the network output (a raw tensor) into the proper
    // function-space format, i.e. B-spline objects for the interior
    // and boundary parts that can be evaluated.
    Base::u_.from_tensor(outputs);

    // Evaluate the Laplacian operator
    auto u_ilapl =
        Base::u_.ilapl(Base::G_, collPts_.first, var_knot_indices_,
                       var_coeff_indices_, G_knot_indices_, G_coeff_indices_);

    auto f =
        Base::f_.eval(collPts_.first, var_knot_indices_, var_coeff_indices_);

    auto u_bdr = Base::u_.template eval<iganet::functionspace::boundary>(
        collPts_.second);

    auto bdr =
        ref_.template eval<iganet::functionspace::boundary>(collPts_.second);

    // Evaluate the loss function
    return torch::mse_loss(*u_ilapl[0], *f[0]) +
           1e1 * torch::mse_loss(*std::get<0>(u_bdr)[0], *std::get<0>(bdr)[0]) +
           1e1 * torch::mse_loss(*std::get<1>(u_bdr)[0], *std::get<1>(bdr)[0]) +
           1e1 * torch::mse_loss(*std::get<2>(u_bdr)[0], *std::get<2>(bdr)[0]) +
           1e1 * torch::mse_loss(*std::get<3>(u_bdr)[0], *std::get<3>(bdr)[0]);

    // SUM of coefficients must be zero (not SUM of U(cpts))!!!
  }
};
    
  }} // namespace iganet::mex

/// @brief Matlab-callable IgANet instance for the two-dimensional
/// Poisson problem with homogeneous Neumann boundary conditions
///
/// This class implements the Matlab-callable Mex function to the
/// IgANet instance for the two-dimensional Poisson problem with
/// homogeneous Neumann boundary conditions
class MexFunction : public matlab::mex::Function {
public:
  /// @brief Call operator
  void operator()(matlab::mex::ArgumentList outputs,
                  matlab::mex::ArgumentList inputs) {

    // Get pointer to engine
    std::shared_ptr<matlab::engine::MATLABEngine> matlabPtr = getEngine();
    
    // Get array factory
    matlab::data::ArrayFactory factory;

    // Check that inputs is not empty
    if (inputs.size() == 0)
      matlabPtr->feval(u"error", 0,
                       std::vector<matlab::data::Array>({factory.createScalar("Input size must not be zero")}));

    // Check that the first input argument is a string
    if (inputs[0].getType() != matlab::data::ArrayType::CHAR &&
        inputs[0].getType() != matlab::data::ArrayType::MATLAB_STRING)
      matlabPtr->feval(u"error", 0,
                       std::vector<matlab::data::Array>({factory.createScalar("First input parameter must be a string")}));

    // Convert to string
    std::u16string cmd = to_u16string(inputs[0]);
    
    if (cmd == u"create") {
      create(outputs, inputs);
    }
    else if (cmd == u"coeffs") {
      matlab::data::TypedArray<double> geoCoeffs =
        factory.createArray<double>({
            static_cast<unsigned long>(net_->G().space().ncumcoeffs()),
            static_cast<unsigned long>(net_->G().space().geoDim())
          });

      for (std::size_t i=0; i<net_->G().space().geoDim(); ++i) {
        auto coeffs = net_->G().space().coeffs(i).template accessor<double, 1>();
        for (std::size_t j=0; j<net_->G().space().ncumcoeffs(); ++j)
          geoCoeffs[j][i] = coeffs[j];
      }

      matlab::data::TypedArray<double> varCoeffs =
        factory.createArray<double>({
            static_cast<unsigned long>(net_->u().space().ncumcoeffs()),
            static_cast<unsigned long>(net_->u().space().geoDim())
          });

      for (std::size_t i=0; i<net_->u().space().geoDim(); ++i) {
        auto coeffs = net_->u().space().coeffs(i).template accessor<double, 1>();
        for (std::size_t j=0; j<net_->u().space().ncumcoeffs(); ++j)
          varCoeffs[j][i] = coeffs[j];
      }

      // Assign the MATLAB arrays to the outputs list
      outputs[0] = geoCoeffs;
      outputs[1] = varCoeffs;
    }
    else if (cmd == u"degrees") {
      matlab::data::TypedArray<double> geoDegrees = factory.createArray<double>({net_->G().space().degrees().size()});
      std::copy(net_->G().space().degrees().cbegin(),
                net_->G().space().degrees().cend(), geoDegrees.begin());

      matlab::data::TypedArray<double> varDegrees = factory.createArray<double>({net_->u().space().degrees().size()});
      std::copy(net_->u().space().degrees().cbegin(),
                net_->u().space().degrees().cend(), varDegrees.begin());

      // Assign the MATLAB arrays to the outputs list
      outputs[0] = geoDegrees;
      outputs[1] = varDegrees;
    }
    else if (cmd == u"eval") {
      
    }
    else if (cmd == u"ncoeffs") {
      matlab::data::TypedArray<double> geoNumCoeffs = factory.createArray<double>({net_->G().space().ncoeffs().size()});
      std::copy(net_->G().space().ncoeffs().cbegin(),
                net_->G().space().ncoeffs().cend(), geoNumCoeffs.begin());

      matlab::data::TypedArray<double> varNumCoeffs = factory.createArray<double>({net_->u().space().ncoeffs().size()});
      std::copy(net_->u().space().ncoeffs().cbegin(),
                net_->u().space().ncoeffs().cend(), varNumCoeffs.begin());

      // Assign the MATLAB arrays to the outputs list
      outputs[0] = geoNumCoeffs;
      outputs[1] = varNumCoeffs;
    }
    else if (cmd == u"release") {
      net_.reset();
    }
    else if (cmd == u"train") {
      
    }
    else
      matlabPtr->feval(u"error", 0,
                       std::vector<matlab::data::Array>({factory.createScalar(u"Invalid first input parameter '" + cmd + u"'")}));    
  }
  
private:
  /// @brief Converts TypedArray<char16_t> to std::u16string
  inline static std::u16string to_u16string(matlab::data::TypedArray<char16_t> array) {
    std::u16string str;
    
    for (const auto& c : array) {
      str += c;
    }
    return str;
  }

  /// @brief Converts std::string to std::u16string
  inline static std::u16string to_u16string(std::string str) {
    std::wstring_convert<std::codecvt_utf8_utf16<char16_t>, char16_t> converter;
    return converter.from_bytes(str);
  }

  /// @brief Creates an IgANets object
  ///
  /// inputs[0] = "create"
  /// inputs[1] = [int, ...]  number of neurons per layer
  /// inputs[2] = [int, int]  number of coefficients of the geometry
  /// inputs[3] = [int, int]  number of coefficients of the variable
  void create(matlab::mex::ArgumentList outputs,
              matlab::mex::ArgumentList inputs) {

    // Get pointer to engine
    std::shared_ptr<matlab::engine::MATLABEngine> matlabPtr = getEngine();
    
    // Get array factory
    matlab::data::ArrayFactory factory;

    try {
    // Check that inputs has four parameters
    if (inputs.size() != 4)
      matlabPtr->feval(u"error", 0,
                       std::vector<matlab::data::Array>({factory.createScalar("Create requires three additional input parameters")}));

    // Get first input parameter
    if (inputs[1].getType() != matlab::data::ArrayType::DOUBLE)
      matlabPtr->feval(u"error", 0,
                       std::vector<matlab::data::Array>({factory.createScalar(u"Invalid second input parameter")}));
    
    std::vector<int64_t> layers;
    for (std::size_t i=0; i< inputs[1].getNumberOfElements(); ++i)
      layers.push_back(inputs[1][i]);

    std::vector<std::vector<std::any>> activations(layers.size(), {iganet::activation::sigmoid} );
    activations.push_back({iganet::activation::none});
    
    // Get second input parameter
    if (inputs[2].getType() != matlab::data::ArrayType::DOUBLE ||
        inputs[2].getNumberOfElements() != 2)
      matlabPtr->feval(u"error", 0,
                       std::vector<matlab::data::Array>({factory.createScalar(u"Invalid third input parameter")}));
    
    std::array<int64_t, 2> geometryMapNumCoeffs{inputs[2][0], inputs[2][1]};

    // Get third input parameter
    if (inputs[3].getType() != matlab::data::ArrayType::DOUBLE ||
        inputs[3].getNumberOfElements() != 2)
      matlabPtr->feval(u"error", 0,
                       std::vector<matlab::data::Array>({factory.createScalar(u"Invalid fourth input parameter")}));

    std::array<int64_t, 2> variableNumCoeffs{inputs[3][0], inputs[3][1]};

    // Create IgANet instance
    net_ = std::make_shared<Net>(layers, activations, geometryMapNumCoeffs, variableNumCoeffs);
    }
    catch(const std::exception& e) {
      matlabPtr->feval(u"error", 0,
                       std::vector<matlab::data::Array>({factory.createScalar(u"The following error occured: " + to_u16string(e.what()))}));
    }
  }
  
  /// @brief IgANet instance type
  using Net = iganet::mex::Poisson<torch::optim::LBFGS,
                                   iganet::S<iganet::UniformBSpline<double, 2, 1, 1>>,
                                   iganet::S<iganet::UniformBSpline<double, 1, 3, 3>>>;

  /// @brief IgANet instance
  std::shared_ptr<Net> net_;
};
